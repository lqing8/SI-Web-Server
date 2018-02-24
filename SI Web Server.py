from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/user/<name>')
def user(name):
    #return '<h1>Hello, %s!</h1>' % name
    return render_template('user.html', name=name)


if __name__ == '__main__':
    app.run()
