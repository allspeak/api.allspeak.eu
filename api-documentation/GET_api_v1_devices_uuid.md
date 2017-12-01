# Device Resources

    GET api/v1/devices/:uuid

## Description
Returns detailed information of a single device.

***

## Requires authentication
* A valid API Key must be provided in **api_key** HTTP header.
***

## Parameters

***

## Return format
A JSON object describing the device in the following format:

- **uuid** — uuid of the device.
- **serial** — Serial of the device
- **manifacturer**
- **model**
- **platform** - The operating system
- **version** - The version of the OS
- **registered_on** — The datetime of the device registration
- **self_url** — The URL of the device

***

## Errors
All known errors cause the resource to return HTTP error code header together with a JSON object containing at least 'status' and 'error' keys describing the source of error.

- **401 Forbidden** — The device cannot be accessed by the current request
- **404 Not Found** — Device with the specified uuid does not exist.


***

## Example
**Request**

    api/v1/devices/4f976f3aabd094ea

**Return** __shortened for example purpose__
``` json
{
    "manufacturer": "Motorola",
    "model": "Moto G Play",
    "platform": "android",
    "registered_on": "2017-11-30 18:07:26",
    "self_url": "https://example.com/api/v1/devices/4f976f3aabd094e",
    "serial": "ZY223QB37R",
    "uuid": "4f976f3aabd094e",
    "version": "6.0.1"
}
```