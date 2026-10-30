"""Hugging Face token handling. The token value is never printed or stored."""
import os

TOKEN_NAME = "my_hf_token"


def get_hf_token():
    """Return the HF token from the environment, then from Colab secrets."""
    tok = os.environ.get(TOKEN_NAME)
    if tok:
        return tok
    try:
        from google.colab import userdata  # type: ignore

        tok = userdata.get(TOKEN_NAME)
    except Exception:
        tok = None
    if not tok:
        raise RuntimeError(
            f"No HF token found. Set env var '{TOKEN_NAME}' or a Colab secret named '{TOKEN_NAME}'."
        )
    return tok


def login_hf():
    """Log in to the Hub without echoing the token."""
    from huggingface_hub import login

    login(token=get_hf_token(), add_to_git_credential=False)
