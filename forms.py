from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, FileField, SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class ResumeChatForm(FlaskForm):
    resume = FileField("Upload Resume (PDF)")
    job_description = TextAreaField("Job Description", validators=[DataRequired()])
    question = StringField("Your Query", validators=[DataRequired()])
    submit = SubmitField("Ask")
