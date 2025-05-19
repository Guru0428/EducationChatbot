from flask import Flask, render_template, request, jsonify, redirect, url_for
from chat import get_response, retrain_model
import sqlite3

app = Flask(__name__)
DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chatbot_reply():
    user_msg = request.args.get("msg")
    tag, reply, confidence = get_response(user_msg)
    if confidence < 0.07:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO unknown_questions (question) VALUES (?)", (user_msg,))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        conn.close()
        reply = "Sorry, I do not understand. Your question has been forwarded to a counselor."
    return jsonify({"tag": tag, "response": reply})

@app.route("/admin")
def admin_dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, question, answer FROM unknown_questions WHERE answer IS NULL")
    unknowns = c.fetchall()
    conn.close()
    return render_template("admin.html", unknowns=unknowns)

@app.route("/admin/answer", methods=["POST"])
def admin_answer():
    qid = request.form.get("id")
    answer = request.form.get("answer")
    if qid and answer:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE unknown_questions SET answer = ? WHERE id = ?", (answer, qid))
        conn.commit()
        conn.close()
        retrain_model()
    return redirect(url_for('admin_dashboard'))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)