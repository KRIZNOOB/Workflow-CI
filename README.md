# Workflow-CI

Project ini menggunakan MLflow Project untuk training ulang model secara otomatis lewat GitHub Actions.

## Struktur

```
Workflow-CI/
	.github/workflows/ci.yml
	MLProject/
		MLProject
		conda.yaml
		modelling.py
		bank_transactions_data_preprocessing.csv
```

## Cara menjalankan lokal

```bash
cd MLProject
mlflow run .
```

Artefak MLflow akan tersimpan di folder `MLProject/mlruns`.

## CI Workflow

Workflow ada di `.github/workflows/ci.yml` dan berjalan saat push ke `main` atau manual trigger.

### Docker Hub (opsional - Advance)

Jika ingin build dan push image Docker lewat CI, buat secret di GitHub repo:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (access token dari Docker Hub)
