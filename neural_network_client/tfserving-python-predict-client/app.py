from flask import Flask, jsonify, make_response, request
from example import main

app = Flask(__name__)



@app.route('/')
def root():
    return 'root'


@app.route('/bert', methods=['GET'])
#@app.route('/bert')
def bert():
    user_string = request.values['comment']
    out = main(user_string)
    return jsonify(out)


if __name__ == '__main__':
    app.run()