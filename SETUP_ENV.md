# Environment Setup Instructions

## ⚠️ Important Security Notice

**NEVER commit actual API keys to version control!**

The `.env.example` file contains placeholders for configuration values. You must create your own `.env` file with actual values.

## Setup Steps

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file and add your actual OpenAI API key:**
   ```bash
   # Open .env in your text editor and replace the placeholder
   OPENAI_API_KEY=your-actual-api-key-here
   ```

3. **Verify .env is in .gitignore:**
   The `.env` file should already be listed in `.gitignore` to prevent accidental commits.

## Configuration Details

### Required Settings

- **OPENAI_API_KEY**: Your OpenAI API key from https://platform.openai.com/api-keys
  - Format: `sk-proj-...` or `sk-...`
  - Required for OpenAI functionality

### Optional Settings

All other settings have sensible defaults but can be customized:

- **API_HOST**: API server host (default: 0.0.0.0)
- **API_PORT**: API server port (default: 8000)
- **DEBUG**: Enable debug mode (default: false)
- **TORCH_DEVICE**: PyTorch device (default: cpu)
- **PYBULLET_GUI**: Enable PyBullet GUI (default: false)

## Testing Your Configuration

After setting up your `.env` file, you can test the configuration:

1. **Check configuration loading:**
   ```bash
   poetry run python -m cogniforge.core.config
   ```

2. **Start the API and test endpoints:**
   ```bash
   # Start the API
   make run-api
   
   # In another terminal, test the configuration endpoint
   curl http://localhost:8000/config
   
   # Test OpenAI connection (requires valid API key)
   curl http://localhost:8000/openai/test
   ```

## Security Best Practices

1. **Never share your `.env` file** - It contains sensitive information
2. **Rotate API keys regularly** - Update them periodically for security
3. **Use different keys for different environments** - Dev, staging, production
4. **Limit API key permissions** - Use the minimum required permissions
5. **Monitor API usage** - Check your OpenAI dashboard for unexpected usage

## Troubleshooting

### OpenAI API Key Not Working

1. Check the key format starts with `sk-`
2. Verify the key is active in your OpenAI dashboard
3. Ensure you have available credits/quota
4. Check for any typos or extra spaces

### Environment Variables Not Loading

1. Ensure `.env` file is in the project root
2. Check file permissions
3. Verify no syntax errors in `.env` file
4. Try explicitly setting the path in your shell:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Python-dotenv Documentation](https://pypi.org/project/python-dotenv/)
- [FastAPI Configuration Guide](https://fastapi.tiangolo.com/advanced/settings/)