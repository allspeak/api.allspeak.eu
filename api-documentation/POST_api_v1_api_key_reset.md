# User Resources

    POST api/v1/api_key_reset

## Description
Reset the **API key** to a new random **API key**.

***

## Requires authentication
* A valid API Key must be provided in **api-key** HTTP header.
***

## Parameters

***

## Return format
A JSON object with the following format:

= **api-key** - The new **API key**

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **401 Forbidden** — Authentication failed


***

## Example
**Request**

    api/v1/api_key_reset

**Data**

**Return** 

``` json
{
	"api-key": "89698084-d5f8-11e7-8b79-0242ac1f0004"
}