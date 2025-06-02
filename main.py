import os
import json
import base64
import asyncio
import re
import time
from urllib.parse import parse_qs, urlparse
from uuid import UUID

import requests
import websockets
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy import text
from starlette.websockets import WebSocketState
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
import pymssql
from websockets import ConnectionClosedError, ConnectionClosedOK

from sql import get_user_by_phone, get_farm, create_db_engine,SQL_QUERY_TEMPLATE,get_mongo_connection_string
from pymongo import MongoClient
from datetime import datetime, timezone,date
from bson import ObjectId
from plivo import *

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
VOICE = 'ballad'
LOG_EVENTS = [
    'session.created', 'session.updated', 'conversation.created',
    'conversation.item.created', 'conversation.item.truncated',
    'conversation.item.deleted', 'conversation.item.input_audio_transcription.completed',
    'conversation.item.input_audio_transcription.failed', 'response.created',
    'response.done', 'response.output_item.added', 'response.output_item.done',
    'response.content_part.added', 'response.content_part.done', 'response.text.delta',
    'response.text.done', 'response.audio_transcript.delta', 'response.audio_transcript.done',
    'response.audio.delta', 'response.audio.done', 'response.function_call_arguments.delta',
    'response.function_call_arguments.done', 'error'
]
active_connections: dict[int, list[WebSocket]] = {}
latest_call_ids: dict[int, set[int]] = {}
app = FastAPI()

# MongoDB connection
# mongo_client = MongoClient(os.getenv('MONGO_URI'))
# mongo_db = mongo_client['devDB']
# call_transcript_collection = mongo_db['CallTranscript']
mongo_client = AsyncIOMotorClient(get_mongo_connection_string())
db = mongo_client['devDB']
call_transcript_collection = db['CallTranscript']


if not OPENAI_API_KEY:
    raise ValueError('Missing OpenAI API key. Set OPENAI_API_KEY in .env file.')

# Global dictionary to store session data by call ID
active_call_sessions = {}


def get_db_connection():
    return pymssql.connect(
        server=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('SQL_USER'),
        password=os.getenv('SQL_PASSWORD'),
        database=os.getenv('SQL_DB'),
        as_dict=True
    )


def extract_caller_id_from_stream_sid(stream_sid):
    """Extract caller identifier from Twilio stream SID"""
    if stream_sid:
        caller_id = stream_sid.replace('MZ', '').replace('CA', '').replace('_', '')
        return caller_id[:10] if len(caller_id) > 10 else caller_id
    return "unknown"



def execute_stored_procedure(cursor, farm_id, action_mode):
    """Execute SP_Bot stored procedure"""
    params = {
        'MasterFarmId': farm_id,
        'ActionMode': action_mode
    }
    cursor.execute("Exec SP_Bot @MasterFarmId=%(MasterFarmId)s, @ActionMode=%(ActionMode)s", params)
    return cursor.fetchall()


def get_soil_report(master_farm_id):
    """Get detailed soil report from database"""
    try:
        query = """
        DECLARE @return_value int;
        EXEC @return_value = [dbo].[SP_SM_Data]
            @MasterFarmId = %s,
            @CropName = NULL,
            @ActionMode = 5;
        """
        with get_db_connection() as cnxn:
            cursor = cnxn.cursor()
            cursor.execute(query, (master_farm_id,))
            results = cursor.fetchall()

            soil_data = {}
            for result in results:
                param_name = result['ParameterName'].lower()
                soil_data[param_name] = {
                    'value': result['ParameterValue'],
                    'unit': result['MasterUnitCode'],
                    'status': result['Remark']
                }
            return soil_data
    except Exception as e:
        print(f"Error in get_soil_report: {str(e)}")
        return {}


def get_weather_data(farm_id):
    """Get latest weather data from SQL Server"""
    try:
        with get_db_connection() as cnxn:
            cursor = cnxn.cursor()
            query = """
                    SELECT TOP 7 ClimateDate, MinTemp, \
                           MaxTemp, \
                           Rainfall, \
                           Humidity,
                           CloudsCover, \
                           CloudStatus, \
                           Windspeed
                    FROM AMClimateDetail
                    WHERE MasterFarmId = %s
                    ORDER BY ClimateDate DESC \
                    """
            cursor.execute(query, (farm_id,))
            return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return []


def get_crop_data(farm_id):
    """Get current crop information"""
    try:
        query = """
        DECLARE @return_value int;
        EXEC @return_value = [dbo].[SP_SM_Data]
            @MasterFarmId = %s,
            @CropName = NULL,
            @ActionMode = 4;
        """
        with get_db_connection() as cnxn:
            cursor = cnxn.cursor()
            cursor.execute(query, (farm_id,))
            results = cursor.fetchall()

            active_operations = [r for r in results if r.get("ProcessStatus") == 1]
            if active_operations:
                return active_operations[0]
            elif results:
                return results[0]
            return {}
    except Exception as e:
        print(f"Error in get_crop_data: {str(e)}")
        return {}


def generate_dynamic_system_message(farm_id):
    """Generate dynamic system message based on database data"""
    try:
        with get_db_connection() as cnxn:
            cursor = cnxn.cursor()
            farm_details = execute_stored_procedure(cursor, farm_id, 1)
            indices_data = execute_stored_procedure(cursor, farm_id, 5)
            govt_schemes = execute_stored_procedure(cursor, farm_id, 8)

        soil_data = get_soil_report(farm_id)
        weather_data = get_weather_data(farm_id)
        crop_data = get_crop_data(farm_id)

        # Build farm details section
        farm_info = ""
        if farm_details:
            farm = farm_details[0]
            farm_info = f"""
FARM DETAILS:
- Name: {farm.get('FarmTitle', 'Unknown')}
- Location: {farm.get('FarmLocation', 'Unknown')}
- Area: {farm.get('FarmArea', 'Unknown')} acres
- Coordinates: {farm.get('FarmCoordinate', 'Unknown')}
- Farm ID: {farm.get('MasterFarmId', farm_id)}
"""

        # Build indices section
        indices_info = "\nCURRENT VEGETATION INDICES:\n"
        if indices_data:
            for idx in indices_data:
                indices_info += f"- {idx.get('IndicesTitle', 'Unknown')}: {idx.get('Mean', 'N/A')} ({idx.get('IndicesCode', 'N/A')})\n"

        # Build soil analysis section
        soil_info = "\nSOIL ANALYSIS:\n"
        if soil_data:
            for param, data in soil_data.items():
                status_indicator = f"({data['status']})" if data['status'] else ""
                soil_info += f"- {param.title()}: {data['value']} {data['unit']} {status_indicator}\n"

        # Build weather section
        weather_info = "\nRECENT WEATHER PATTERN:\n"
        if weather_data:
            latest_weather = weather_data[0]
            weather_info += f"Latest recorded on {latest_weather.get('ClimateDate', 'Unknown date')}: "
            weather_info += f"Temp: {latest_weather.get('MinTemp', 'N/A')}-{latest_weather.get('MaxTemp', 'N/A')}°C, "
            weather_info += f"Humidity: {latest_weather.get('Humidity', 'N/A')}%, "
            weather_info += f"Rainfall: {latest_weather.get('Rainfall', 'N/A')}mm, "
            weather_info += f"Wind: {latest_weather.get('Windspeed', 'N/A')}km/h\n"

        # Build crop information
        crop_info = "\nCURRENT CROP INFORMATION:\n"
        if crop_data:
            crop_info += f"- Crop: {crop_data.get('CropTitle', 'Unknown')}\n"
            crop_info += f"- Growth Stage: {crop_data.get('CropLifeCycleTitle', 'Unknown')}\n"
            if crop_data.get('OperationTitle'):
                crop_info += f"- Current Operation: {crop_data.get('OperationTitle')}\n"

        # Build government schemes section
        schemes_info = "\nAVAILABLE GOVERNMENT SCHEMES:\n"
        if govt_schemes:
            for scheme in govt_schemes:
                schemes_info += f"- {scheme.get('GovtSchemeTitle', 'Unknown Scheme')}\n"

        system_message = f"""
You are Agripilot, an AI farming assistant for Indian farmers with access to real-time farm data. 

IMPORTANT COMMUNICATION GUIDELINES:
1. LANGUAGE HANDLING: Farmers may speak in Hindi, English, Marathi, or mix these languages. 
   - Always respond in the SAME language mix that the farmer uses
   - If they ask in Hindi, respond in Hindi
   - If they mix languages, mix languages in your response
   - Common Hindi/Marathi words: "mere khet ka", "mera farm", "soil kaisa hai", "paani dena chahiye kya"

2. CONTEXT UNDERSTANDING:
   - When farmers ask "mera farm ID kya hai" or "what is my farm ID", they mean FARM ID, not phone ID
   - Farm ID is specifically the agricultural land identification number: {farm_id}
   - Phone ID और Farm ID अलग होते हैं - don't confuse them

3. RESPONSE STYLE:
   - Be conversational and friendly like talking to a local farmer
   - Use terms farmers understand: "khet", "fasal", "mititi", "paani"
   - Give practical, actionable advice
   - Include units that farmers understand (acres, kg, temperature in Celsius)

{farm_info}
{indices_info}
{soil_info}
{weather_info}
{crop_info}
{schemes_info}

SPECIFIC RESPONSES FOR COMMON QUESTIONS:
- Farm ID questions: "Aapka farm ID hai {farm_id}"
- Soil health: Mention specific nitrogen/phosphorus levels and recommendations
- Weather impact: Relate to current farming activities
- Government schemes: Explain in simple terms how to apply

INSTRUCTIONS:
- Provide specific, practical advice based on the above data
- Address soil deficiencies and suggest appropriate fertilizers
- Consider weather impacts on farming decisions  
- Recommend relevant government schemes when applicable
- Keep responses conversational and helpful for Indian farmers
- Focus on actionable recommendations
- Never mention being an AI or having limitations
- Always base advice on the provided farm-specific data
- Match the language style of the farmer (Hindi/English/Marathi mix)
"""

        return system_message.strip()

    except Exception as e:
        print(f"Error generating dynamic system message: {str(e)}")
        return f"""
You are Agripilot, an AI farming assistant for Farm ID {farm_id}. 
You can communicate in Hindi, English, and Marathi as needed.
When farmers ask about their farm ID, tell them it's {farm_id}.
Provide practical agricultural advice and recommendations based on farming best practices.
Keep responses helpful and action-oriented for Indian farmers.
"""


@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": "Agripilot Dynamic Voice Assistant is running!", "status": "active"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    try:
        form_data = await request.form()
        caller_phone = form_data.get("From", "unknown")
        call_id = form_data.get("CallUUID", f"call_{caller_phone}")

        # Get user details
        user_details = await get_user_by_phone(caller_phone)

        if not user_details:
            response = plivoxml.ResponseElement()
            response.add(plivoxml.SpeakElement(
                "Sorry, we couldn't find your registered number. Please register yourself at Agripilot.AI. "
                "If you need any assistance, feel free to reach out. Thank you for calling, and have a great day!",
                voice="Polly.Salli",
                language="en-US"
            ))
            return HTMLResponse('<?xml version="1.0" encoding="UTF-8"?>\n' + response.to_string(),
                                media_type="application/xml")

        # Get farm data
        user_id = user_details.get("MasterLoginId")
        farm_list = await get_farm(user_id)
        selected_farm = farm_list[0]
        farm_id = selected_farm['MasterFarmId']

        # Store session data for WebSocket access
        active_call_sessions[call_id] = {
            'user_details': user_details,
            'farm_id': farm_id,
            'caller_phone': caller_phone,
            'call_start_time': datetime.now()
        }

        print(f"Stored session data for call {call_id} with farm_id {farm_id}")

        # Create TwiML response to connect to WebSocket
        response = plivoxml.ResponseElement()

        response.add(plivoxml.SpeakElement(
            f"Hello! Welcome to AgriPilot AI. I'm ready to assist you. Please speak after the beep.",
            voice="Polly.Salli",
            language="en-US"
        ))
        response.add(plivoxml.WaitElement(length=1))

        wss_host = os.getenv('HOST_URL')

        # Pass call_id as a URL parameter for WebSocket
        stream = response.add(plivoxml.StreamElement(
            f"{wss_host}/media-stream?call_id={call_id}",
            bidirectional=True,
            streamTimeout=86400,
            keepCallAlive=True,
            contentType="audio/x-mulaw;rate=8000",
            audioTrack="inbound"
        ))

        return HTMLResponse('<?xml version="1.0" encoding="UTF-8"?>\n' + response.to_string(),
                            media_type="application/xml")

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


async def save_conversation_transcript(call_details, role, message):
    """Save conversation transcript in the specified format with multiple message fields."""
    try:
        from datetime import datetime

        # Use await for async MongoDB operations
        existing_conversation = await call_transcript_collection.find_one({
            "token_id": call_details['token_id'],
            "farmer_id": call_details['farmerMasterLoginId'],
            "caller_id": call_details['callID']
        })

        current_time = datetime.utcnow()

        if existing_conversation:
            message_count = 0
            for key in existing_conversation.keys():
                if key.startswith('message'):
                    if key == 'message':
                        message_count = 1
                    elif '_' in key:
                        try:
                            num = int(key.split('_')[1])
                            message_count = max(message_count, num + 1)
                        except:
                            pass
                    else:
                        message_count = max(message_count, 1)

            if message_count == 0:
                message_field = "message"
                role_field = "role"
            else:
                message_field = f"message_{message_count}"
                role_field = f"role_{message_count}"

            # Use await for update operation
            result = await call_transcript_collection.update_one(
                {"_id": existing_conversation["_id"]},
                {
                    "$set": {
                        message_field: message,
                        role_field: role,
                        "updated_at": current_time
                    }
                }
            )
            print(f"Updated conversation with {message_field}: {message[:50]}...")

        else:
            conversation_doc = {
                "token_id": call_details['token_id'],
                "farmer_id": call_details['farmerMasterLoginId'],
                "caller_id": call_details['callID'],
                "role": role,
                "message": message,
                "created_at": current_time
            }

            # Use await for insert operation
            result = await call_transcript_collection.insert_one(conversation_doc)
            print(f"Created new conversation: {result.inserted_id}")

    except Exception as e:
        print(f"Error saving conversation transcript: {e}")
        import traceback
        traceback.print_exc()



@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Main WebSocket handler for Plivo-OpenAI communication."""
    await websocket.accept()
    query_params = parse_qs(websocket.url.query)
    call_id = query_params.get('call_id', [None])[0]

    if not call_id or call_id not in active_call_sessions:
        print(f"Invalid or missing call_id: {call_id}")
        await websocket.close(code=1000, reason="Invalid call session")
        return

    # Get session data
    session_data = active_call_sessions[call_id]
    farm_id = session_data['farm_id']
    user_details = session_data['user_details']

    print(f"WebSocket connected for call {call_id} with farm_id {farm_id}")
    # Generate dynamic system message with fresh database data
    dynamic_system_message = generate_dynamic_system_message(farm_id)
    print("Generated dynamic system message with current farm data")

    openai_ws = None
    disconnect_called = False  # Flag to prevent multiple disconnect calls
    connection_active = True  # Track connection state

    # Connection state
    stream_id = None
    caller_id = None
    latest_media_timestamp = 0
    last_assistant_item = None
    mark_queue = []
    response_start_timestamp = None
    current_user_transcript = ""
    current_assistant_response = ""
    call_details = {}  # Store API response details

    async def call_disconnect_api():
        """Call the disconnect API when the call ends."""
        nonlocal disconnect_called, connection_active
        if disconnect_called:
            print("Disconnect API already called, skipping...")
            return

        disconnect_called = True
        connection_active = False
        token_id = call_details.get('token_id')
        GO_BACKEND_URL = os.getenv('GO_BACKEND_URL')

        if token_id:
            try:
                response = requests.patch(
                    f"{GO_BACKEND_URL}/disconnectCallByAgronomist?tokenID={token_id}",
                    timeout=5
                )
                print(f"Disconnect API called successfully: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Failed to call disconnect API for token {token_id}: {e}")
        else:
            print("No token_id available for disconnect API call")

    async def monitor_websocket_connection():
        """Monitor WebSocket connection and detect disconnections"""
        try:
            while connection_active and not disconnect_called:
                try:
                    # For FastAPI WebSocket, we can't use ping(), so we'll use a different approach
                    # Check if the WebSocket is still connected by checking its state
                    if websocket.client_state != WebSocketState.CONNECTED:
                        print("WebSocket connection lost - calling disconnect API")
                        await call_disconnect_api()
                        break

                    # Alternative: Try to send a small control frame or check connection health
                    # We can try sending a tiny message or just rely on the connection state check
                    await asyncio.sleep(5)  # Check every 5 seconds

                except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                    print("WebSocket connection lost - calling disconnect API")
                    await call_disconnect_api()
                    break
                except Exception as e:
                    print(f"WebSocket connection check failed: {e} - calling disconnect API")
                    await call_disconnect_api()
                    break
        except asyncio.CancelledError:
            print("Connection monitor cancelled")
            if not disconnect_called:
                await call_disconnect_api()

    try:
        async with websockets.connect(
                f"{os.getenv('AZURE_OPENAI_API_ENDPOINT')}",
                extra_headers={
                    "api-key": f"{os.getenv('AZURE_OPENAI_API_KEY')}",
                }
        ) as openai_ws:

            # Initialize OpenAI session with dynamic farm data
            await initialize_session(openai_ws, dynamic_system_message)

            async def receive_from_plivo():
                """Process audio from Plivo and forward to OpenAI."""
                nonlocal stream_id, latest_media_timestamp, caller_id, call_details, connection_active
                try:
                    async for message in websocket.iter_text():
                        if not connection_active:
                            break

                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON received: {e}")
                            continue

                        # Handle Plivo media events
                        if data['event'] == 'media' and openai_ws and openai_ws.open:
                            # Plivo sends audio in different format than Twilio
                            latest_media_timestamp = int(data.get('sequenceNumber', 0))
                            audio_data = data['media']['payload']

                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": audio_data
                            }
                            await openai_ws.send(json.dumps(audio_append))

                        elif data['event'] == 'start':
                            # Plivo stream start event
                            stream_id = data['start']['streamId']
                            caller_id = data['start'].get('callId', 'unknown')
                            print(f"Voice session started: {stream_id}, Caller: {caller_id}")

                            # Hit the API when stream starts
                            try:
                                GO_BACKEND_URL = os.getenv('GO_BACKEND_URL')
                                response = requests.post(
                                    f'{GO_BACKEND_URL}/connectToAgenticAI',
                                    json={"farmerMasterLoginId": user_details.get('MasterLoginId')},
                                    headers={'Content-Type': 'application/json'},
                                    timeout=10
                                )
                                response_data = response.json()

                                token_id = response_data.get('details', {}).get('tokenID')
                                farmer_id = response_data.get('details', {}).get('farmerMasterLoginId')
                                api_call_id = response_data.get('details', {}).get('callID')

                                if not all([token_id, farmer_id, api_call_id]):
                                    print(
                                        f"Missing fields: token_id={token_id}, farmer_id={farmer_id}, call_id={api_call_id}")
                                else:
                                    call_details['token_id'] = token_id
                                    call_details['farmerMasterLoginId'] = farmer_id
                                    call_details['callID'] = api_call_id
                                    print(
                                        f"API call successful. Token ID: {token_id}, Farmer ID: {farmer_id}, Call ID: {api_call_id}")

                            except requests.exceptions.RequestException as e:
                                print(f"API call failed: {e}")
                            except Exception as e:
                                print(f"Error in API call: {e}")

                            # Send initial message to OpenAI to start conversation
                            if openai_ws and openai_ws.open:
                                await openai_ws.send(json.dumps({
                                    "type": "input_audio_buffer.commit"
                                }))

                        elif data['event'] == 'stop':
                            print("Plivo stream stopped - user hung up")
                            await call_disconnect_api()
                            break

                        elif data['event'] == 'mark':
                            # Handle Plivo mark events
                            if mark_queue:
                                mark_queue.pop(0)

                        # Handle any other Plivo-specific disconnect events
                        elif data['event'] in ['hangup', 'disconnect', 'end']:
                            print(f"Call ended via {data['event']} event")
                            await call_disconnect_api()
                            break

                except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                    print("WebSocket connection closed by client - calling disconnect API")
                    await call_disconnect_api()
                except asyncio.CancelledError:
                    print("Receive task cancelled - calling disconnect API")
                    await call_disconnect_api()
                    raise
                except Exception as e:
                    print(f"Error in receive_from_plivo: {e}")
                    await call_disconnect_api()

            async def send_to_plivo():
                """Process OpenAI responses and send audio to Plivo."""
                nonlocal stream_id, last_assistant_item, response_start_timestamp
                nonlocal current_user_transcript, current_assistant_response, caller_id, call_details, connection_active

                try:
                    async for openai_message in openai_ws:
                        if not connection_active:
                            break

                        response = json.loads(openai_message)

                        if response['type'] in LOG_EVENTS:
                            print(f"Event: {response['type']}")

                        # Handle user speech transcription
                        if response.get('type') == 'conversation.item.input_audio_transcription.completed':
                            transcript = response.get('transcript', '').strip()
                            if transcript and call_details:
                                current_user_transcript = transcript
                                print(f"User said: {transcript}")
                                await save_conversation_transcript(call_details, "user", transcript)

                        # Handle assistant text response
                        if response.get('type') == 'response.text.delta':
                            text_delta = response.get('delta', '')
                            current_assistant_response += text_delta

                        # Handle assistant audio transcript
                        if response.get('type') == 'response.audio_transcript.delta':
                            transcript_delta = response.get('delta', '')
                            current_assistant_response += transcript_delta

                        # Handle when assistant starts responding
                        if response.get('type') == 'response.created':
                            current_assistant_response = ""

                        # Handle when assistant finishes responding
                        if response.get('type') == 'response.audio_transcript.done':
                            transcript = response.get('transcript', '').strip()
                            if transcript and call_details:
                                print(f"Assistant responded (audio transcript): {transcript}")
                                await save_conversation_transcript(call_details, "assistant", transcript)
                                current_assistant_response = ""

                        # When response is complete, save the full assistant message
                        # if response.get('type') == 'response.done':
                        #     if current_assistant_response.strip() and call_details:
                        #         print(f"Assistant responded: {current_assistant_response}")
                        #         save_conversation_transcript(call_details, "assistant",
                        #                                      current_assistant_response.strip())
                        #         current_assistant_response = ""

                        # Handle audio response from OpenAI
                        if response.get('type') == 'response.audio.delta' and 'delta' in response:
                            try:
                                # Check if WebSocket is still connected before sending
                                if websocket.client_state == WebSocketState.CONNECTED:
                                    audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode(
                                        'utf-8')
                                    audio_delta = {
                                        "event": "playAudio",
                                        "media": {
                                            "contentType": 'audio/x-mulaw',
                                            "sampleRate": 8000,
                                            "payload": audio_payload
                                        }
                                    }
                                    await websocket.send_json(audio_delta)

                                    if response_start_timestamp is None:
                                        response_start_timestamp = latest_media_timestamp

                                    if response.get('item_id'):
                                        last_assistant_item = response['item_id']

                                    await send_mark(websocket, stream_id)
                                else:
                                    print("WebSocket not connected - stopping audio transmission")
                                    await call_disconnect_api()
                                    break

                            except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                                print("WebSocket closed while sending audio - calling disconnect API")
                                await call_disconnect_api()
                                break
                            except Exception as e:
                                print(f"Error sending audio to websocket: {e}")
                                await call_disconnect_api()
                                break

                        # Handle interruptions
                        if response.get('type') == 'input_audio_buffer.speech_started':
                            print("Farmer started speaking - handling interruption")
                            await handle_interruption()

                except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                    print("OpenAI WebSocket connection closed - calling disconnect API")
                    await call_disconnect_api()
                except asyncio.CancelledError:
                    print("Send task cancelled - calling disconnect API")
                    await call_disconnect_api()
                    raise
                except Exception as e:
                    print(f"Error in OpenAI communication: {e}")
                    await call_disconnect_api()

            async def handle_interruption():
                """Handle when farmer interrupts the AI response."""
                nonlocal response_start_timestamp, last_assistant_item

                if mark_queue and response_start_timestamp and last_assistant_item:
                    elapsed_time = latest_media_timestamp - response_start_timestamp

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    if openai_ws and openai_ws.open:
                        await openai_ws.send(json.dumps(truncate_event))

                    # Plivo clear event
                    try:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "event": "clear",
                                "streamId": stream_id
                            })
                    except:
                        pass  # Don't fail interruption handling if websocket is closed

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp = None

            async def send_mark(connection, stream_id):
                """Send timing marks for audio synchronization."""
                if stream_id and connection.client_state == WebSocketState.CONNECTED:
                    try:
                        mark_event = {
                            "event": "mark",
                            "streamId": stream_id,
                            "mark": {"name": "responsePart"}
                        }
                        await connection.send_json(mark_event)
                        mark_queue.append('responsePart')
                    except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
                        print("WebSocket closed while sending mark")
                        await call_disconnect_api()
                    except Exception as e:
                        print(f"Error sending mark: {e}")

            # Run all tasks concurrently including connection monitoring
            try:
                await asyncio.gather(
                    receive_from_plivo(),
                    send_to_plivo(),
                    monitor_websocket_connection(),
                    return_exceptions=True
                )
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                await call_disconnect_api()

    except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
        print("Main WebSocket connection closed - calling disconnect API")
        await call_disconnect_api()
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        await call_disconnect_api()
    finally:
        # Ensure disconnect API is called when WebSocket closes
        print("WebSocket connection closing - ensuring disconnect API is called")
        if not disconnect_called:
            await call_disconnect_api()


async def initialize_session(openai_ws, system_message):
    """Initialize OpenAI session with dynamic farm-specific instructions."""
    enhanced_system_message = f"""
    {system_message}

    IMPORTANT LANGUAGE INSTRUCTIONS:
    - Always respond in the same language that the user speaks
    - If the user speaks in English, respond ONLY in English
    - If the user speaks in Hindi, respond ONLY in Hindi
    - Detect the language from the user's speech and match it exactly
    - Do not mix languages in your response
    - Default to English if language detection is unclear
    """
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {"model": "whisper-1"},
            "voice": VOICE,
            "instructions": enhanced_system_message,
            "modalities": ["text", "audio"],
            "temperature": 0.7,
        }
    }

    print('Initializing Agripilot session with dynamic farm data and transcription')
    await openai_ws.send(json.dumps(session_update))
active_clients = {}

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sid = str(id(websocket))
    print(f"Client connected: {sid}")
    active_clients[sid] = {"socket": websocket, "filter": None}

    try:
        while True:
            data = await websocket.receive_json()
            print(f"Received filter from {sid}: {data}")

            # Save filter
            filter_data = {
                "farmer_id": data.get("farmer_id"),
                "token_id": data.get("token_id"),
                "caller_id": data.get("caller_id")
            }
            active_clients[sid]["filter"] = filter_data

            # Fetch existing chat history
            existing = await call_transcript_collection.find_one(filter_data)
            if existing:
                messages = parse_conversation(existing)
                await websocket.send_json({"type": "initial", "messages": messages})

    except Exception as e:
        print(f"Client {sid} disconnected: {e}")
    finally:
        del active_clients[sid]


# Convert complex message format into array of dicts
def parse_conversation(doc):
    results = []
    created_time = doc.get("created_at")
    timestamp = (
        created_time.isoformat() if isinstance(created_time, datetime) else str(created_time)
    )

    for i in range(0, 50):  # Assuming max 50 messages per doc
        msg = doc.get(f"message_{i}") if f"message_{i}" in doc else None
        role = doc.get(f"role_{i}") if f"role_{i}" in doc else None
        if msg and role:
            results.append({
                "message": msg,
                "sender_type": role,
                "timestamp": timestamp
            })

    # Add root-level message if present
    if doc.get("message") and doc.get("role"):
        results.insert(0, {
            "message": doc["message"],
            "sender_type": doc["role"],
            "timestamp": timestamp
        })

    return results


# MongoDB Change Stream Listener
async def watch_changes():
    async with call_transcript_collection.watch(full_document="updateLookup") as stream:
        async for change in stream:
            doc = change.get("fullDocument")
            if not doc:
                continue

            # Match all connected clients by filter
            for sid, client_data in active_clients.items():
                ws = client_data["socket"]
                filters = client_data["filter"]
                if not filters:
                    continue

                if (
                    doc.get("farmer_id") == filters["farmer_id"] and
                    doc.get("token_id") == filters["token_id"] and
                    doc.get("caller_id") == filters["caller_id"]
                ):
                    payload = {
                        "type": change["operationType"],
                        "messages": parse_conversation(doc)
                    }
                    await ws.send_json(payload)

# def convert_datetime_to_str(data_list):
#     for row in data_list:
#         for key, value in row.items():
#             if isinstance(value, (datetime, date)):
#                 row[key] = value.isoformat()
#     return data_list
def convert_datetime_to_str(data_list):
    for row in data_list:
        for key, value in row.items():
            if isinstance(value, (datetime, date)):
                row[key] = value.isoformat()
            elif isinstance(value, UUID):
                row[key] = str(value)
    return data_list

async def fetch_call_logs(login_id: int):
    engine = create_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text(SQL_QUERY_TEMPLATE), {"login_id": login_id})
        rows = result.fetchall()
        columns = result.keys()
        return [dict(zip(columns, row)) for row in rows]

async def call_log_watcher(login_id: int):
    while login_id in active_connections:
        new_data = await fetch_call_logs(login_id)
        current_ids = {row['CallID'] for row in new_data}

        if login_id not in latest_call_ids or latest_call_ids[login_id] != current_ids:
            latest_call_ids[login_id] = current_ids
            for websocket in active_connections[login_id]:
                await websocket.send_text(json.dumps({"type": "update", "data": convert_datetime_to_str(new_data)}))
        await asyncio.sleep(5)

@app.websocket("/ws/call-logs/{login_id}")
async def websocket_endpoint(websocket: WebSocket, login_id: int):
    await websocket.accept()

    if login_id not in active_connections:
        active_connections[login_id] = []
        asyncio.create_task(call_log_watcher(login_id))

    active_connections[login_id].append(websocket)
    try:
        initial_data = await fetch_call_logs(login_id)
        latest_call_ids[login_id] = {row['CallID'] for row in initial_data}
        await websocket.send_text(json.dumps({"type": "init", "data": convert_datetime_to_str(initial_data)}))

        while True:
            await websocket.receive_text()  # just keep alive

    except WebSocketDisconnect:
        active_connections[login_id].remove(websocket)
        if not active_connections[login_id]:
            del active_connections[login_id]
            latest_call_ids.pop(login_id, None)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(post_startup_task())
    asyncio.create_task(watch_changes())  # MongoDB watcher


async def post_startup_task():
    await asyncio.sleep(0.5)  # Ensures this runs after startup
    # await test_ai_voice_assistant_connection() # Test AI voice assistant connection


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)