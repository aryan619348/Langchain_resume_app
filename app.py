import os
from flask import Flask, render_template, request, redirect
from embeddings_functions import pdf_embeddings

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
    return redirect("/")


@app.route("/delete/<filename>")
def delete(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
