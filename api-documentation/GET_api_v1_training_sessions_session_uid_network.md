# Training Session Resources

    GET api/v1/training_sessions/:session_uid/network

## Description
Returns the trained network for the session

***

## Requires authentication
* A valid API Key must be provided in **api_key** HTTP header.
* An guest training session can be accessed without providing an API key.
***

## Parameters

***

## Return format

A binary object with the trained network

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **401 Forbidden** — The training session cannot be accessed by the current request
- **404 Not Found** — Training session with the specified uid does not exist.

***

## Example
**Request**

    api/v1/training_sessions/323b5468-d684-11e7-999f-0242ac1f0003/network

**Return** __shortened for example purpose__

    Binary object