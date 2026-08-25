from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, FileField, SubmitField
from wtforms.validators import DataRequired, Optional


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class ResumeChatForm(FlaskForm):
    resume = FileField("Upload Resume (PDF)")
    job_description = TextAreaField("Job Description", validators=[DataRequired()])
    question = StringField("Your Query", validators=[Optional()])
    submit = SubmitField("Ask")
