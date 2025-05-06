from flask_login import UserMixin

users = {"siddhi": {"password": ""}}


class User(UserMixin):
    def __init__(self, id):
        self.id = id
