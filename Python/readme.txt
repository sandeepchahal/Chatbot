This is the sample chatbot application which is specifically created for Hospitals or related fields. 
It start with creating embedding for txt file given at specific location.
It also maintain the history chat as well.
The process is as follows:
 . User type a message, it will be first checked in history
    . if Similar repsond found, it will be given to OPenAI with question and answer to rephrase it as per question.
    . If no history response found, it would check in the context i.e. embedding file.
      . Similar results would be send which needs to be filtered based on the token size.
      . Next, context with limited token along with question would be given to OpenAI to extract the information and respond
    . If no content is found from history as well as document, Api call will be made to ChatGPT to get the correspond answer.
During these steps, context from document and history, total token count would be calculated and reduced if nececssary. This is because there is a limitation on prompt in openAI. 
