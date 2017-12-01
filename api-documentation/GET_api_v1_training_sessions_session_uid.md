# Training Session Resources

    GET api/v1/training_sessions/:session_uid

## Description
Returns detailed information of training_session

***

## Requires authentication
* A valid API Key must be provided in **api_key** HTTP header.
* An guest training session can be accessed without providing an API key.
***

## Parameters

***

## Return format

A JSON object describing the training session in the following format:

- **status** - A status indicating if the training_session has been completed or it's still pending
*TODO*

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **401 Forbidden** — The training session cannot be accessed by the current request
- **404 Not Found** — Training session with the specified uid does not exist.

***

## Example
**Request**

    api/v1/training_sessions/323b5468-d684-11e7-999f-0242ac1f0003

**Return** __shortened for example purpose__
``` json
{
    "commands": [
        {
            "id": 1102,
            "title": "Sono arrabbiato"
        },
        {
            "id": 1103,
            "title": "Sono triste"
        },
        {
            "id": 1105,
            "title": "Voglio stare solo"
        },
        {
            "id": 1202,
            "title": "Ho dolore"
        },
        {
            "id": 1203,
            "title": "Ho caldo"
        },
        {
            "id": 1204,
            "title": "Ho freddo"
        },
        {
            "id": 1205,
            "title": "Ho fame"
        },
        {
            "id": 1206,
            "title": "Ho sete"
        },
        {
            "id": 1207,
            "title": "Ho sonno"
        },
        {
            "id": 1208,
            "title": "Devo andare in bagno"
        },
        {
            "id": 1209,
            "title": "Voglio cambiare posizione"
        },
        {
            "id": 1210,
            "title": "Ho troppa saliva"
        },
        {
            "id": 1211,
            "title": "Ho bisogno di lavarmi"
        },
        {
            "id": 1302,
            "title": "Passami il telefono"
        },
        {
            "id": 1305,
            "title": "Chiama ..."
        },
        {
            "id": 1306,
            "title": "Salutami ..."
        },
        {
            "id": 1400,
            "title": "Grazie"
        },
        {
            "id": 1401,
            "title": "Prego"
        },
        {
            "id": 1404,
            "title": "Ciao"
        },
        {
            "id": 1405,
            "title": "Ci vediamo domani"
        },
        {
            "id": 1407,
            "title": "Tutto bene?"
        },
        {
            "id": 1616,
            "title": "Come stai?"
        },
        {
            "id": 1619,
            "title": "Ti voglio bene"
        }
    ],
    "nContextFrames": 5,
    "nInputParams": 792,
    "nItemsToRecognize": 23,
    "nModelType": 275,
    "nProcessingScheme": 252,
    "sCreationTime": "2017/12/01 10:45:57",
    "sInputNodeName": "inputs/I",
    "sLabel": "test",
    "sLocalFolder": "test",
    "sModelFileName": "optimized_user_ft_323b5468-d684-11e7-999f-0242ac1f0003_252.pb",
    "sOutputNodeName": "SMO",
    "status": "complete"
}
```