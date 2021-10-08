from flask import Flask, request, jsonify
app = Flask(__name__) 
@app.route("/all_crypto",methods=['GET'])
