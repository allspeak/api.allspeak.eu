# Device Resources

    POST api/v1/devices

## Description
Creates a new device or update the already existing one with the input **uuid**.

***

## Requires authentication
* A valid API Key must be provided in **api_key** HTTP header.
***

## Parameters
A JSON object describing the device in the following format:

- **uuid** — ID of the device.
- **serial** — Serial of the device
- **manifacturer**
- **model**
- **platform** - The operating system
- **version** - The version of the OS
- **self_url** — The URL of the device

***

## Return format
A http redirect with status code 201 to the URL of the device *api/v1/devices/:**uuid***

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **401 Forbidden** — Authentication failed


***

## Example
**Request**

    api/v1/devices

**Data**

``` json
{
	"uuid": "4f976f3aabd094e",
	"model": "Moto G Play",
	"manufacturer": "Moto",
	"platform": "android",
	"serial": "ZY223QB37R",
	"version": "6.0.1"
	
}
```

**Return** __shortened for example purpose__

    HTTP redirect to *https://example.com/api/v1/devices/4f976f3aabd094e*