
from openai import OpenAI
import os
import pandas as pd
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
pd.options.mode.copy_on_write = True

client = OpenAI(
    api_key= os.environ.get("OPENAI_API_KEY")
)
split_sentence_filename = 'sentences.csv'
maximum_tokens = 300
embeddings_list = []

class MyEmbedding:    
    def _remove_newlines(self,line):
        line = line.str.replace('\n', ' ')
        line = line.str.replace('\\n', ' ')
        line = line.str.replace('  ', ' ')
        line = line.str.replace('  ', ' ')
        return line
    
    def create_embeding_csv_file(self, directory_path,input_csv_filename,output_embedding_filename, model ="text-embedding-ada-002" ):
        try:
            print(f'Reading from CSV file from {os.path.join(directory_path,input_csv_filename)} & Creating Embedding')
            file_data = pd.read_csv(os.path.join(directory_path,input_csv_filename))
            columns = file_data.columns.tolist()
            combined_text =  file_data[columns].apply(lambda row: ' '.join(map(str, row)), axis=1)
            df = pd.DataFrame(combined_text, columns = ['text'])
            df['text'] = self._remove_newlines(df.text)
            texts_list = df['text'].tolist()
            embed = client.embeddings.create(input=texts_list,model=model)
            embeddings_list = [item.embedding for item in embed.data]
            df['embeddings'] = embeddings_list
            df.to_csv(directory_path+output_embedding_filename)
            return True
        except Exception as e:
            print('ERROR while creating embedding from CSV File \n',e)
            raise Exception("an error occurred")
    