"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt no formato username/nome
        prompt_data: Dados do prompt vindos do YAML

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        system_prompt = prompt_data.get("system_prompt", "")
        user_prompt = prompt_data.get("user_prompt", "{bug_report}")
        description = prompt_data.get("description", "")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        print(f"   Fazendo push para: {prompt_name}")

        hub.push(
            prompt_name,
            prompt_template,
            new_repo_is_public=True,
            new_repo_description=description,
        )

        print(f"   ✓ Push realizado com sucesso")
        print(f"   ✓ Disponível em: https://smith.langchain.com/prompts/{prompt_name.split('/')[1]}")
        return True

    except Exception as e:
        print(f"   ❌ Erro ao fazer push: {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    return validate_prompt_structure(prompt_data)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS PARA O LANGSMITH")

    required_vars = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]
    if not check_env_vars(required_vars):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    yaml_path = "prompts/bug_to_user_story_v2.yml"

    print(f"Carregando prompt de: {yaml_path}")
    data = load_yaml(yaml_path)
    if not data:
        return 1

    prompt_key = "bug_to_user_story_v2"
    prompt_data = data.get(prompt_key)
    if not prompt_data:
        print(f"❌ Chave '{prompt_key}' não encontrada no YAML")
        return 1

    print("Validando prompt...")
    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1
    print("✓ Prompt válido")

    prompt_name = f"{username}/{prompt_key}"
    success = push_prompt_to_langsmith(prompt_name, prompt_data)

    if success:
        print("\n✅ Push concluído com sucesso!")
        print(f"\nPróximo passo: python src/evaluate.py")
        return 0

    print("\n❌ Falha no push do prompt.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
