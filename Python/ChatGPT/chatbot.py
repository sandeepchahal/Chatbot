from openai import OpenAI
import pandas as pd
import os
from myEmbedding import MyEmbedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
 
prompt_max_token= 3500
response_max_token = 500

file_directory_name = "content/"
current_dir = os.path.dirname(os.path.abspath(__file__))
content_directory = os.path.join(current_dir, file_directory_name)

embedding_file_name = "embeddings.csv"
csv_inputput_file_name = "patients.csv"

not_applicable_response = "Sorry, I can only answer the questions that are relevant to the medical or health care"

default_message = {
    "role": "system",
    "content": (
        "You are an assistant in a Apollo Hospital. "
        "Answer the questions precisely by following below rules: \n"
        "1. Evaluate the question, and if it doesn't pertain to subjects such as medicine, viruses, health, diseases, doctors, patients information or hospitals, say'NAAA'. \n"
        "2. Only answer to valid questions. \n"
        "3. Do not justify your answers. \n"
        "4. Respond appropriately to greetings. \n"+
        "5. Response appropriately to user introductions\n"
    )
}

default_context_message = {
    "role": "system",
    "content": (  
        "6. The 'Context' is having the patients information such as name, address, illness, date,etc. You should answer correctly when asked about patients information"
        "7. Answer questions only based on the provided 'Context'. \n"
        "8. If the given 'Context' does not provide any information about question, only say 'NF'. \n"
        "9. Never provide information that is not mentioned in the given 'Context'. \n"
    )
}

embedding=MyEmbedding()
context_message,api_message_context, user_history = [],[],[]

context_message.append(default_message)
context_message.append(default_context_message)
api_message_context.append(default_message)
    
OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPEN_API_KEY)

def main():
    global context_df, columns_list
    ## load file into variable

    ## creating embedding from CSV file
    if os.path.exists(content_directory + embedding_file_name) == False:
        if embedding.create_embeding_csv_file(directory_path=content_directory, input_csv_filename=csv_inputput_file_name, output_embedding_filename=embedding_file_name):
            print("Embedding is created")
        else:
            print("Embedding not created")
    else:
        print("Embedding already exists")
    
    original_file = pd.read_csv(os.path.join(content_directory,csv_inputput_file_name )) ## original file with patient info
    columns_list = original_file.columns.tolist()


    # Creating embedding from text Files
    ## Only 1 would work 
    # if os.path.exists(content_directory +embedding_file_name ) == False:
    #     if embedding.create_embedding_text_files(directory_path=content_directory, output_csv_filename=embedding_file_name):
    #         print("Embedding is created")
    #     else:
    #         print("Embedding not created")
    # else:
    #     print("Embedding already exists")
        
    context_df = pd.read_csv(os.path.join(content_directory,embedding_file_name))

    context_df['embeddings'] = context_df['embeddings'].apply(ast.literal_eval)

    demo = gr.ChatInterface(answer_question)
    demo.launch() 

def search_response_from_embedding(user_input):
    print('seaching in Document')
    global context_df
    embeddings_array = np.array(list(context_df['embeddings']))
   
    question_embedding = client.embeddings.create(input=user_input, model='text-embedding-ada-002').data[0].embedding

    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(embeddings_array)

    # Find the indices of the nearest neighbors
    _, indices = nn_model.kneighbors([question_embedding])

    # Extract the relevant information from the DataFrame based on the indices
    similar_responses_df =  context_df.iloc[indices[0]]

    # Process the relevant information, e.g., extract the text or other attributes
    return process_similar_responses(similar_responses_df, 2000)
    
def process_similar_responses(similar_responses_df, max_tokens):
    print(similar_responses_df['text'].tolist())

    responses = similar_responses_df['text'].tolist()
    
    # Initialize variables
    cumulative_tokens = 0
    selected_responses = []

    # Iterate through responses until the token limit is reached
    for response in responses:
        response_tokens = len(tokenizer.encode(response))
        # Check if adding the response exceeds the token limit
        if cumulative_tokens + response_tokens <= max_tokens:
            selected_responses.append(response)
            cumulative_tokens += response_tokens
        else:
            break  # Stop adding responses when the limit is reached
    total_count_tokens = 0
    for resp in selected_responses:
        total_count_tokens += len(tokenizer.encode(resp))
        
    print('Token Count of selected responses in Document - ',total_count_tokens)
    return selected_responses

def document_answer(question, history):
    try:
        # Create a chat comxpletion using the question and context
        global columns_list
        context = search_response_from_embedding(question)
        print('context from document', context)

        final_context_reduced = reduce_context_size_if_needed(context)
        final_context = context_message.copy() # it will always keep the copy of default context along with context created from file
        if(len(history)>0):
            reduce_history_if_needed(history)
            final_context.extend(history)

        if(len(columns_list)>0):
            input_data = {'columns': columns_list, 'data': final_context_reduced}
            final_context.append({"role":"user", "content": f"Context: {input_data}\n\n---\n\nQuestion: {question}"})
        else:
            final_context.append({"role":"user", "content": f"Context: {final_context_reduced}\n\n---\n\nQuestion: {question}"})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=final_context,
            temperature=0,
            max_tokens=response_max_token,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        #print('Error In Document Response \n',e)
        return "error" 

def reduce_history_if_needed(history_context,max_tokens = 1500):
    total_tokens = 0
    if(len(history_context)==0):
        return history_context
    
    for message in history_context:
        total_tokens+= len(tokenizer.encode(message['content']))
        total_tokens+= len(tokenizer.encode(message['content']))
    print('history token-count before--------------',total_tokens)
    
    if(total_tokens>max_tokens):
        while(total_tokens> prompt_max_token):
            history_context.pop(0) # Remove the oldest message if the total tokens exceed the limit
            count =0
            for message in history_context:
                count+= len(tokenizer.encode(message[0]))
                count+= len(tokenizer.encode(message[1]))
            total_tokens = count
            
    print('history token-count After --------------',total_tokens)
    
def reduce_context_size_if_needed(context, max_tokens=1500):
    total_tokens = 0
    for message in context:
        total_tokens+= len(tokenizer.encode(message))
    
    print('token-count before',total_tokens,"\n")
    
    if(total_tokens>max_tokens):
        while(total_tokens> max_tokens):
            context.pop(0) # Remove the oldest message if the total tokens exceed the limit
            count =0
            for message in context:
                count+= len(tokenizer.encode(message))
            total_tokens = count
            
    total_tokens = 0
    for message in context:
        total_tokens+= len(tokenizer.encode(message))
    print('token-count after',total_tokens,"\n")
        
    return context

def find_answer_from_history(history,question):
    similarity_threshold= 0.6
   # Extract all user messages from the history
    user_messages = [message["content"] for message in history if message.get("role") == "user" and message.get("content") is not None]

    # Add the current user question to the list
    user_messages.append(question)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_messages)

    # Calculate cosine similarities between the current question and all previous questions
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the index of the most similar question
    most_similar_index = similarities.argmax()
    # Check if the similarity is above the threshold
    if similarities[0, most_similar_index] >= similarity_threshold:
        return history[most_similar_index+1]['content'] # return the answer which would index+1 
    else:
        return 'notfound'

def history_answer(question, history):
    try:
        # Create a chat comxpletion using the question and context
        if(len(history)==0):
            return 'no-history'
        
        history_answer = find_answer_from_history(history=history,question=question)
        print('context from history', history_answer)
        
        if(history_answer =="notfound"):
            return "no-history"
        
        user_message = {"role": "user", "content": f"{question}"}
        answer_message = {"role": "assistant", "content": f"{history_answer}"}
        
        reduce_history_if_needed(history)
        conversion_context= api_message_context.copy() # always add the context along with conversion history context that we save in the file
        conversion_context.extend(history)
    
        conversion_context.append(user_message)
        conversion_context.append(answer_message)
        
        conversion_context.append({"role":"user", "content": f"{question}"})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversion_context,
            temperature=0,
            max_tokens=response_max_token,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print('Error In History Response \n',e)
        return "error" 

def add_conversation(question,answer,history):
    userMessage = {"role": "user", "content": f"{question}"}
    answer = {"role": "assistant", "content": f"{answer}"}

    history.append(userMessage)
    history.append(answer)

def answer_question(question, history):
    try:
        history_ans= history_answer(history=user_history, question=question)
        
        print('Cache: -',history_ans)
        if(history_ans == "error"):
            return "An error has occurred while processing the request"  
        if(history_ans !="no-history"):
            return history_ans
        
        document_ans = document_answer(question=question, history=user_history)
        print('Document response -'+document_ans+"\n")

        if(document_ans == "error"):
            return "An error has occurred while processing the request" 
         
        answer =''
        if(document_ans!="NAAA" and document_ans != '' and document_ans !="NF"):
            answer= document_ans
        elif(document_ans == "NAAA"):
            return not_applicable_response
        else:
            # look for the answer from cloud
            print('Searching on cloud')
            reduce_history_if_needed(user_history)
            conversion_history=api_message_context.copy()
            if(len(user_history)>0):
                conversion_history.extend(user_history)
                
            user_message = {"role": "user", "content": f"{question}"}
            conversion_history.append(user_message)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages= conversion_history,
                temperature=0,
                max_tokens=response_max_token,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        
            answer = response.choices[0].message.content
            print("API response - ",answer)
        if(answer == "NAAA"):
            return f"{not_applicable_response}"
        elif(answer == "NF"):
            return f"Couldn't find the relevant answer"
        # save the correct response to history
        add_conversation(question=question,answer=answer,history= user_history)
        return answer
    except Exception as e:
        print(e)
        return "An error has occurred while processing the request" 

if __name__ == "__main__":
    main()
    





