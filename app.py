import os
from flask import Flask, render_template, request, redirect
from embeddings_functions import pdf_embeddings, url_embeddings
from process import process_answer
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"


@app.route("/")
def index():
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("upload.html", files=files)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    pdf_embeddings(file_path)
    # Process uploaded file
    return redirect("/")

@app.route("/submit-link", methods=["POST"])
def submit_link():
    global job_post_link
    job_post_link = request.form.get("job-link")  # Get the Job Post Link from the form
    url_embeddings(job_post_link)
    return redirect("/")

@app.route("/delete/<filename>")
def delete(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect("/")

@app.route("/chat", methods=["POST"])
def chat():
    global job_post_link
    question = request.form.get("question")
    response = process_answer(question)
    return response

if __name__ == "__main__":
    app.run(debug=True)
