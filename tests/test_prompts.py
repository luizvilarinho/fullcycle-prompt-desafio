"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt_data():
    data = load_prompts(PROMPT_FILE)
    return data[PROMPT_KEY]


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data
        assert prompt_data["system_prompt"].strip() != ""

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt_data["system_prompt"].lower()
        role_markers = ["você é um", "você é uma", "voce e um", "voce e uma"]
        assert any(marker in system_prompt for marker in role_markers), (
            "Nenhuma definição de persona (ex: 'Você é um Product Manager...') "
            "encontrada no system_prompt"
        )

    def test_prompt_mentions_format(self, prompt_data):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt_data["system_prompt"].lower()
        format_markers = [
            "como um", "eu quero", "para que",
            "user story", "critérios de aceitação",
        ]
        assert any(marker in system_prompt for marker in format_markers), (
            "Prompt não menciona o formato padrão de User Story "
            "(ex: 'Como um... eu quero... para que...')"
        )

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt_data["system_prompt"].lower()

        input_markers = system_prompt.count("bug reportado")
        output_markers = system_prompt.count("user story gerada")

        assert input_markers >= 2 and output_markers >= 2, (
            "Esperados pelo menos 2 pares de exemplo entrada/saída "
            "('Bug reportado' / 'User Story gerada') para caracterizar Few-shot Learning"
        )

        techniques = [t.lower() for t in prompt_data.get("techniques_applied", [])]
        assert any("few-shot" in t or "few shot" in t for t in techniques), (
            "Técnica 'Few-shot Learning' não declarada em techniques_applied"
        )

    def test_prompt_no_todos(self, prompt_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        full_text = prompt_data["system_prompt"] + prompt_data.get("user_prompt", "")
        assert "[TODO]" not in full_text
        assert "TODO" not in full_text

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied", [])
        assert len(techniques) >= 2, (
            f"Esperado no mínimo 2 técnicas em techniques_applied, "
            f"encontradas: {len(techniques)}"
        )

    def test_prompt_passes_structure_validation(self, prompt_data):
        """Verifica que o prompt passa na validação estrutural usada pelo push_prompts.py."""
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Prompt inválido: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
