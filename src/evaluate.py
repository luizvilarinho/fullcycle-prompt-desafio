"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados como experimento no LangSmith (visível em Datasets & Experiments)
7. Exibe resumo no terminal com os outputs gerados

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from langsmith.schemas import Run, Example
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()

# Pontuação mínima exigida em todas as métricas (ver IA/context.md)
PASSING_THRESHOLD = 0.8


def get_llm():
    return get_configured_llm(temperature=0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")
        prompt = hub.pull(prompt_name)
        print(f"   ✓ Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando: {prompt_name}")

    prompt_template = pull_prompt_from_langsmith(prompt_name)
    llm = get_llm()
    chain = prompt_template | llm

    example_counter = [0]

    def target(inputs: dict) -> dict:
        example_counter[0] += 1
        n = example_counter[0]
        bug = inputs.get("bug_report", "N/A")

        print(f"\n{'─' * 60}")
        print(f"[{n}/15] BUG:\n{bug}")

        try:
            response = chain.invoke(inputs)
            answer = response.content
            print(f"\nUSER STORY GERADA:\n{answer}")
            print(f"{'─' * 60}")
            return {"answer": answer}
        except Exception as e:
            print(f"      ⚠️  Erro ao gerar resposta: {e}")
            return {"answer": ""}

    def f1_evaluator(run: Run, example: Example) -> dict:
        answer = (run.outputs or {}).get("answer", "")
        reference = (example.outputs or {}).get("reference", "")
        question = (example.inputs or {}).get("bug_report", "")
        if not answer:
            print(f"   F1: 0.00 (sem resposta)")
            return {"key": "f1_score", "score": 0.0}
        result = evaluate_f1_score(question, answer, reference)
        score = result["score"]
        print(f"   F1: {score:.2f}")
        return {"key": "f1_score", "score": score}

    def clarity_evaluator(run: Run, example: Example) -> dict:
        answer = (run.outputs or {}).get("answer", "")
        reference = (example.outputs or {}).get("reference", "")
        question = (example.inputs or {}).get("bug_report", "")
        if not answer:
            print(f"   Clarity: 0.00 (sem resposta)")
            return {"key": "clarity", "score": 0.0}
        result = evaluate_clarity(question, answer, reference)
        score = result["score"]
        print(f"   Clarity: {score:.2f}")
        return {"key": "clarity", "score": score}

    def precision_evaluator(run: Run, example: Example) -> dict:
        answer = (run.outputs or {}).get("answer", "")
        reference = (example.outputs or {}).get("reference", "")
        question = (example.inputs or {}).get("bug_report", "")
        if not answer:
            print(f"   Precision: 0.00 (sem resposta)")
            return {"key": "precision", "score": 0.0}
        result = evaluate_precision(question, answer, reference)
        score = result["score"]
        print(f"   Precision: {score:.2f}")
        return {"key": "precision", "score": score}

    experiment_name = prompt_name.split("/")[-1]
    print(f"\n   Experimento será registrado em: '{dataset_name}' no LangSmith")
    print("   Iniciando avaliação...\n")

    try:
        results = ls_evaluate(
            target,
            data=dataset_name,
            evaluators=[f1_evaluator, clarity_evaluator, precision_evaluator],
            experiment_prefix=experiment_name,
            client=client,
            max_concurrency=1,
        )

        all_results = list(results)

        f1_scores = []
        clarity_scores = []
        precision_scores = []

        print(f"\n\n{'=' * 50}")
        print("Resumo por exemplo:")
        for i, r in enumerate(all_results, 1):
            ex_f1 = ex_clarity = ex_precision = 0.0
            for er in r.get("evaluation_results", {}).get("results", []):
                score = er.score if er.score is not None else 0.0
                if er.key == "f1_score":
                    ex_f1 = score
                    f1_scores.append(score)
                elif er.key == "clarity":
                    ex_clarity = score
                    clarity_scores.append(score)
                elif er.key == "precision":
                    ex_precision = score
                    precision_scores.append(score)
            print(f"   [{i:02d}] F1:{ex_f1:.2f} Clarity:{ex_clarity:.2f} Precision:{ex_precision:.2f}")

        if not f1_scores:
            return {"helpfulness": 0.0, "correctness": 0.0, "f1_score": 0.0, "clarity": 0.0, "precision": 0.0}

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return {"helpfulness": 0.0, "correctness": 0.0, "f1_score": 0.0, "clarity": 0.0, "precision": 0.0}


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Derivadas:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=PASSING_THRESHOLD)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=PASSING_THRESHOLD)}")

    print("\nMétricas Base:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=PASSING_THRESHOLD)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=PASSING_THRESHOLD)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=PASSING_THRESHOLD)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    all_above_threshold = all(score >= PASSING_THRESHOLD for score in scores.values())
    passed = all_above_threshold and average_score >= PASSING_THRESHOLD

    if passed:
        print(f"\n✅ STATUS: APROVADO - Todas as métricas >= {PASSING_THRESHOLD}")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        failed_metrics = [name for name, score in scores.items() if score < PASSING_THRESHOLD]
        if failed_metrics:
            print(f"⚠️  Métricas abaixo de {PASSING_THRESHOLD}: {', '.join(failed_metrics)}")
        print(f"⚠️  Média atual: {average_score:.4f} | Necessário: {PASSING_THRESHOLD:.4f}")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")

    # Garante que os traces vão para o projeto correto
    os.environ["LANGCHAIN_PROJECT"] = project_name

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        print("   Configure seu username do LangSmith Hub antes de continuar.")
        return 1

    prompts_to_evaluate = [
        f"{username}/bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "helpfulness": 0.0,
                    "correctness": 0.0,
                    "f1_score": 0.0,
                    "clarity": 0.0,
                    "precision": 0.0
                },
                "passed": False
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print(f"✅ Todos os prompts atingiram todas as métricas >= {PASSING_THRESHOLD}!")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print(f"\n✓ Experimento registrado em:")
        print(f"  https://smith.langchain.com/datasets/{dataset_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print(f"⚠️  Alguns prompts não atingiram todas as métricas >= {PASSING_THRESHOLD}")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1

if __name__ == "__main__":
    sys.exit(main())
