import os
import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description='Upload index.json to Hugging Face hub')
    parser.add_argument('--index', default='index.json', help='Local index file')
    parser.add_argument('--repo', required=True, help='Destination repo-id, e.g. username/dataset')
    parser.add_argument('--path-in-repo', default='index.json', help='Path inside the repo')
    args = parser.parse_args()

    token = os.getenv('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN environment variable not set')

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=args.index,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type='dataset',
    )
    print(f'Uploaded {args.index} to {args.repo}/{args.path_in_repo}')


if __name__ == '__main__':
    main()
