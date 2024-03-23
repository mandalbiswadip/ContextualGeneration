import openai

openai.api_key = ""

def get_chat_completion(user_message, system_message=None, model="gpt-3.5-turbo"):
    messages = []
    if system_message is not None:
        messages.append( {"role": "system", "content": system_message})

    messages.append( {"role": "user", "content": user_message})

    completion = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message
