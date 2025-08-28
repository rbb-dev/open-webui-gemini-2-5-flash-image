# Gemini 2.5 Flash Image Pipe for Open WebUI

This is a custom pipe for [Open WebUI](https://openwebui.com/) that integrates with the Gemini 2.5 Flash model via the Google generateContent API.
It allows users to generate new images from text prompts or edit existing images by providing a prompt and an image from the chat history.

- **ID**: gemini_flash_img
- **Author**: rbb-dev ([GitHub](https://github.com/rbb-dev/))
- **Version**: 0.1.5

## Features
- Generate images from text descriptions.
- Edit images by uploading or referencing the last image in the chat, combined with a text prompt.
- Uploads generated images to the Open WebUI file library for easy access.
- Supports streaming and non-streaming responses.
- Configurable via valves for API key, base URL, and request timeout.

## Requirements
- Open WebUI installed and running.
- A valid API key for the Gemini API (or compatible service, e.g., via CometAPI proxy as in the code).

## Installation
Open WebUI allows importing pipes directly from a URL. Follow these steps:

1. In your Open WebUI admin interface, navigate to the Pipes section (or equivalent for adding custom pipes/models).

2. Use the "Import from Link" feature and provide the URL to the raw `open-webui-gemini-2-5-flash-image.py` file from this repository. For example:
https://github.com/rbb-dev/open-webui-gemini-2-5-flash-image/raw/refs/heads/main/open-webui-gemini-2-5-flash-image.py

3. Once imported, configure the valves (e.g., API key) in Open WebUI.

## Configuration
Configure the pipe via the valves in Open WebUI's admin interface or directly in the code:

- **COMET_API_KEY**: Your API key for the Gemini service (required).
- **API_BASE_URL**: Base URL for the API (default: `https://api.CometAPI.com/v1beta` – update if using a different endpoint, e.g., official Google API).
- **REQUEST_TIMEOUT**: Timeout for API requests in seconds (default: 600).

<img width="831" height="1753" alt="image" src="https://github.com/user-attachments/assets/742d464f-ed22-439f-a27c-13de15723bf3" />
