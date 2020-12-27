import os, time
from flask import Flask, render_template, request, jsonify
import fruit_pred

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["DEBUG"] = 0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.form.get("img-data")
        data = data.split(',')
        data = [int(num) for num in data]
        with open("static/image/new_image","wb") as f:
            f.write(bytearray(data))
        response = fruit_pred.predict()
        return render_template("index.html", response=response)
    return render_template("index.html")


#<<<<<<< HEAD
#@app.route("/prediction", methods=["GET","POST"])
@app.route("/prediction", methods=["POST"])
#def prediction():
#=======
#	@app.route("/prediction", methods=["POST"])
def prediction():
    # if request.method == "GET":
        # return render_template("predict.html")  # Case testing
#>>>>>>> c26bff1c97825da4a14ac457ae0b7a1645583f36
    data = request.form.get("img-data")
    data = data.split(',')
    data = [int(num) for num in data]
    with open("static/image/new_image","wb") as f:
        f.write(bytearray(data))
    result = fruit_pred.predict() + 1
    return render_template("predict.html", prediction=result)
#<<<<<<< HEAD
#=======

if __name__== "__main__":
    app.run()
#>>>>>>> c26bff1c97825da4a14ac457ae0b7a1645583f36
