from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field,ValidationError,ValidatorFunctionWrapHandler
import logging
from typing import Generic,Optional,Union,Type

load_dotenv()
openai_key=os.getenv("OPENAI_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openai_key
)
model = "mistralai/mistral-small-3.2-24b-instruct:free"

class EventType(BaseModel):
    """First LLM call to extract basic info and determine type of event"""
    description : str = Field(description = "raw info regarding the query")
    is_email : bool = Field(description="check if request made by user is an email request")
    confidence_mail : float = Field(description="confidence that request is an email request. A value between 0 and 1")
    is_msg : bool = Field(description="does the request have a messaging component?")
    confidence_msg : float = Field(description="confidence that request is a messaging request. A value between 0 and 1")
    has_datetime : bool = Field(description="does it have a time and date for sending the mail?")
    has_ph : bool = Field(description="does request have phone number to send draft to on telegram?")
    has_mail_ID : bool = Field(description="does the request have a mail ID?")

EventType_example = EventType(
    description="user wants to send mail at 4 pm on 10 July 2025 to aashrith2002@gmail.com telling him to attend a meeting at 9 pm on Wednesday.send draft of mail to +919945498228 on Telegram.",
    is_email=True,
    confidence_mail=0.99,
    is_msg=False,
    confidence_msg=0.87,
    has_datetime=True,
    has_ph =True,
    has_mail_ID = True
)

class EventDetails(BaseModel):
    """Second LLM call to parse details"""
    recepient_email_address : str = Field(description="this is the email address of the recepient")
    date_time : str = Field(description="date and time for when the event must happen. Example : If task is to send email at 8 pm on Tuesday," 
    "then time would be 8 pm and date would be the date of closest upcoming Tuesday from current date. Use ISO 8601 Date and time format")
    ph_num : str = Field(description="phone number to which draft of mail needs to be sent via telegram")

# DATETIME TO BE ADDED.
EventDetails_example = EventDetails(
    recepient_email_address="aashrith2002@gmail.com",
    date_time ="",
    ph_num ="+919945498282"
)

class GeneratedMailDraft(BaseModel):
    """Third LLM call to generate draft of the mail"""
    mail_subject : str = Field(description="subject of the mail based on extracted details")
    mail_body : str = Field(description="body of the mail based on extracted details")

GeneratedMailDraft_example = GeneratedMailDraft(
mail_subject = "Attend Meeting",
mail_body = """Hello Vijay.

Please attend meeting at 9 pm on Wednesday.

Thank you.
Yours sincerely,
Sai Aashrith"""
)

    
class EventApproval(BaseModel):
    """Fourth LLM call to accept user feedback"""
    approval : bool = Field(description="Describve the state of user's approval. True if user accepts draft without feeback, else false.")
    changes : str = Field(description="changes suggested by user")

class Email(BaseModel):
    """Format for Gmail API payload"""
    email : str = Field(description="the email in RFC 5322 format") 

class TelegramUser(BaseModel):
    """This object represents a Telegram user or bot."""
    id : int = Field(description="id of user")
    is_bot : Optional[bool] = Field(description="boolean to identify whether user is a bot")
    first_name : Optional[str] = Field(description="First name of user")
    last_name : Optional[str] = Field(description="last name of user")
    username : Optional[str] = Field(description="username of user")

class TelegramChat(BaseModel):
    """This object represents a chat."""
    id : int = Field(description="id of chat")
    type : str = Field(description="Type of the chat, can be either “private”, “group”, “supergroup” or “channel”")
    title : Optional[str] = Field(description="title of group")
    username : Optional[str] = Field(description="username of chat")
    first_name : Optional[str] = Field(description="first name of other person if private chat")
    last_name : Optional[str] = Field(description="last name of other person if private chat")
    
class TelegramMessage(BaseModel):
    """This object represents a message."""
    message_id : int = Field(description="id of message inside a chat")
    message_thread_id : Optional[int] = Field(description="Unique identifier of a message thread to which the message belongs; for supergroups only")
    from_ : Optional[TelegramUser] = Field(description="sender of the message",alias="from")
    chat : TelegramChat = Field(description="Chat the message belongs to")
    date : int = Field(description="Date the message was sent in Unix time. It is always a positive number, representing a valid date.")
    text : Optional[str] = Field(description="For text messages, the actual UTF-8 text of the message")

class TelegramUpdate(BaseModel):
    """This object represents an incoming update."""
    update_id : int = Field(description="unique id of update")
    message : Optional[TelegramMessage] = Field(description="new incoming message")


class SendMessageRequest(BaseModel):
    """Schema for request to send message to user using sendMessage method.
       This request is made to the telegram server which returns a response with confirmation of the message it sent. Telegram generates the msg 
       based on details we give it in this request."""
    chat_id : Union[int,str] = Field(description="Unique identifier for target chat")
    text : str = Field(description="Text of the msg to be sent")
    parse_mode : Optional[str] = Field(description="Mode for parsing entities in the message text.")
    # reply_markup -> this is for inline/reply keyboard. add  this feature later.

class TelegramAPIResponse(BaseModel):
    """This is the confirmation response that Telegram sends after we send msg request"""
    ok : bool = Field(description="specifies whether the request was successful or not")
    result : Optional[TelegramMessage] = Field(description="if request was successful, the result appears here")
    error_code : Optional[int] = Field(description="if there is an error, the error code is specifed here")
    description : Optional[str] = Field(description="if there is any error, it is described here")

def make_structured_prompt(pydantic_class :Type[BaseModel], example_obj :Type[BaseModel], task_description: str) -> str:
    """Helper function to write prompts"""
    return f"""
    You are an assistant.Your task is {task_description}.
    Return ONLY valid JSON of the following format:
    {pydantic_class}.
    This is an example:
    {example_obj.model_dump_json(indent=2)}
    Do not return any extra text, only the json as given in the example. 
    """

def output_validator(fn,required_ouput :Type[BaseModel],max_retries:int) -> Optional[BaseModel]:
    """Utility function to validate model output and retry if needed"""
    for attempt in range(max_retries):
        try:
            result =fn() 
        except Exception as e:
            print(f"{fn} failed with exception :{e} at attempt {attempt+1}")
            continue
            
        try:
            return required_output.model_validate_json(result)
        except ValidationError as ve:
            print(f"Validation error :{ve} at attempt {attempt+1}")
            continue
    raise Exception("Failed to get valid output after retries. Explore more for exact error.")

task_desc_CheckEmailRequest = "Extract basic info and determine type of query.Confirm if query involves sending a mail,message and has required details like email ID,phone number,and date and time details."
prompt_CheckEmailRequest = make_structured_prompt(EventType,EventType_example,task_desc_CheckEmailRequest)

def CheckEmailRequest(query : str) -> str:
    """LLM call to figure out whether the asked task is accurate"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role":"system",
                "content":prompt_CheckEmailRequest
            },
            {
                "role":"user",
                "content":query
            },
        ],
    )
    result_CheckEmailRequest = completion.choices[0].message.content
    return result_CheckEmailRequest

task_desc_GetDetails = "Extract email address,phone number and datetime from the query."
prompt_GetDetails = make_structured_prompt(EventDetails,EventDetails_example,task_desc_GetDetails)

def GetDetails(query : str) -> str:
    """LLM call to extract details from query"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role" : "system",
                "content" : prompt_GetDetails
            },
            {
                "role" : "user",
                "content" : query
            },
        ],
    )
    result_GetDetails = completion.choices[0].message.content
    return result_GetDetails

task_desc_DraftMail = "Draft a mail subject and body based on query"
prompt_DraftMail = make_structured_prompt(GeneratedMailDraft,GeneratedMailDraft_example,task_desc_DraftMail)

def DraftMail(query : str)-> str:
    """LLM call to generate mail subject and body from query"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role" : "system",
                "content" : prompt_DraftMail
            },
            {
                "role" : "user",
                "content" : query
            },
        ],
    )
    result_DraftMail = completion.choices[0].message.content
    return result_DraftMail

def SuggestChangesBasedOnFeedback(query : str) -> str:




                                                                                            
                                                                                    