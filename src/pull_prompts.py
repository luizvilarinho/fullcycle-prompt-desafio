"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith():
    print_section_header("PULL DE PROMPTS DO LANGSMITH")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return False

    prompt_name = "leonanluppi/bug_to_user_story_v1"
    output_path = "prompts/bug_to_user_story_v1.yml"

    print(f"Fazendo pull do prompt: {prompt_name}")

    try:
        prompt = hub.pull(prompt_name)
        print("✓ Prompt carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao fazer pull: {e}")
        return False

    system_prompt = ""
    user_prompt = ""

    for message in prompt.messages:
        if isinstance(message, SystemMessagePromptTemplate):
            system_prompt = message.prompt.template
        elif isinstance(message, HumanMessagePromptTemplate):
            user_prompt = message.prompt.template

    prompt_data = {
        "bug_to_user_story_v1": {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "created_at": "2025-01-15",
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    if save_yaml(prompt_data, output_path):
        print(f"✓ Prompt salvo em: {output_path}")
        return True

    print(f"❌ Erro ao salvar prompt")
    return False


def main():
    """Função principal"""
    success = pull_prompts_from_langsmith()
    if success:
        print("\n✅ Pull concluído com sucesso!")
        return 0
    print("\n❌ Falha no pull dos prompts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
