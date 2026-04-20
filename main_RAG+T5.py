import nest_asyncio
nest_asyncio.apply()
import os
import logging
import csv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss

from huggingface_hub import login
login()

import os
os.environ["HF_TOKEN"] = "***"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=300, overlap=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

os.environ["TELEGRAM_TOKEN"] = "***"
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH", "messages.csv")

with open("/content/tabular_data.txt", "r", encoding="utf-8") as f:
    table_text = f.read()

with open("/content/text_data.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = chunk_text(full_text, chunk_size=50)



embeddings = embedder.encode(chunks, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def make_prompt(x, chat):
  q_emb = embedder.encode([chat], convert_to_numpy=True)
  D, I = index.search(q_emb, 7)
  retrieved_chunks = [chunks[i] for i in I[0]]
  context = "\n\n".join(retrieved_chunks)
  return f"""
** Suppose you are Munir El Haddadi. Always the question in the **first person** (using "I", "my")**

--Follow these rules when answering:
-If someone says "hello" or "hi", you must only answer with the word "hello".
-If someone asks "how are you", you must only answer with "I'm fine."
-*** Only use information from the section called 'Information sections'. ***
-If the information is not in the Information section, say "I don't remember that."
-Keep answers short and natural (max 2–3 sentences).


**Information sections:**

My favorite team is FC Barcelona.
{x}

{context}

Question: {chat}

** Answer as Munir El Haddadi: **
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/flan-t5-large'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def save_message_to_csv(user_id, username, user_message, bot_reply=None):
    file_exists = os.path.isfile(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['user_id', 'username', 'user_message', 'bot_reply'])
        writer.writerow([user_id, username, user_message, bot_reply or ""])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, This is Munir!")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("tap \\start")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    chat_id = update.effective_chat.id
    username = update.effective_user.username or update.effective_user.first_name

    if len(text) > 2000:
        await update.message.reply_text("too long")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        prompt = make_prompt(table_text, text)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                do_sample=True,
                temperature=1.0,
            )
        model_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        '''
        separator = "Answer as Munir El Haddadi:"
        if separator in full_output:
            model_text = full_output.split(separator, 1)[1].strip()
        else:
            model_text = full_output.strip()
        '''
        save_message_to_csv(chat_id, username, text, model_text)

        if len(model_text) <= 4000:
            await update.message.reply_text(model_text)
        else:
            bio = BytesIO(model_text.encode("utf-8"))
            bio.name = "response.txt"
            await context.bot.send_document(chat_id=chat_id, document=bio, filename="response.txt")

    except Exception as e:
        logger.exception("Unhandled error")
        await update.message.reply_text("inner error?!" + str(e))


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN!!!")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("polling")
    app.run_polling()

if __name__ == "__main__":
    main()
