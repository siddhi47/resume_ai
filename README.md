# resume_ai

Ever got tired of filling out questionnaires while applying for jobs? This resume helps you create a resume in seconds. Just provide your resume and it will generate answers to common questions while filling out a job.

You need these things:

- Your resume in PDF format
- OpenAI API key
- docker-compose installed.

Before you think to yourself, 'Oh no. OpenAI API key? That has to be expensive?' Well, it's cheaper than subscribing to tools like Simplify or ChatGPT Plus. And you can use it for other things too. So, it's not a waste of money. With just 5 dollars you can generate hundreds and even thousands of answers. So, it's worth it.

## Step 1: Get a resume in PDF format

You can use your own resume or use a template. You can find templates on websites like Canva, Google Docs, or Microsoft Word. Just make sure to save it in PDF format.
I like this <a href = "https://www.overleaf.com/latex/templates/rendercv-engineeringresumes-theme/shwqvsxdgkjy"> template</a>.

## Step 2: Get OpenAI API key

Just create an account on OpenAI and get your API key. You can find it in the API section of your account. Just copy it and paste it in the .env file.

Note: in the same .env file there is a key named SECRET_KEY. This is used to encrypt the data. You can use any random string for this. Just make sure to keep it secret. You can use a password manager to generate a random string. Just make sure it's at least 16 characters long.

## Step 3: Install docker-compose

You can find the installation instructions for docker-compose <a href = "https://docs.docker.com/compose/install/">here</a>. Just follow the instructions for your operating system. It's pretty easy to install. Just make sure you have Docker installed. You can find the installation instructions for Docker <a href = "https://docs.docker.com/get-docker/">here</a>.

## Step 4: Run the app

You can run the app using docker-compose. Just run the following command in the root directory of the project:

```bash
docker-compose up
```

### Additional steps

Add password to /src/resume_bot/auth.py
This is not the best way to store your password. But it works for development. You can use a password manager to generate a random string. Just make sure it's at least 16 characters long.

## TODOS

- Add more prompts
- Use better way to store password for flask app.
- Add CICD pipeline

