from flask_login import UserMixin

users = {"siddhi": {"password": "password123"}}


class User(UserMixin):
    def __init__(self, id):
        self.id = id
