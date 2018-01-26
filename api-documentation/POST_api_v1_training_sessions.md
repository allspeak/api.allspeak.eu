# Training Session Resources

    POST api/v1/training_sessions

## Description
Create a new training_session

***

## Requires authentication
* A valid API Key must be provided in **api-key** HTTP header.
* An guest training session can be created without providing an API key.
***

## Parameters

A ZIP file with the training information (See the [samples folder](../web/project/samples))

***

## Return format

A JSON object describing the training session in the following format:

- **session_uid** - The uid of the training session

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **400 Invalid request** — The input data are not correct

***

## Example
**Request**

    api/v1/training_sessions

**Data**

    ZIP file

**Return** __shortened for example purpose__
``` json
{
    "session_uid": "323b5468-d684-11e7-999f-0242ac1f0003"
}
```