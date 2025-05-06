from flask_login import UserMixin

users = {"siddhi": {"password": "482c811da5d5b4bc6d497ffa98491e38"}}


class User(UserMixin):
    def __init__(self, id):
        self.id = id
