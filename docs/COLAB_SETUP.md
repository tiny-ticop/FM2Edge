# Colab setup for the private repository

The FM2Edge repository can remain private. Colab reads a repository-scoped
GitHub token from its Secrets panel and uses it only while cloning or pulling.
The token is not placed in the notebook, Git URL, repository configuration, or
cell output.

## 1. Create a read-only fine-grained token

Open GitHub **Settings > Developer settings > Personal access tokens >
Fine-grained tokens**, then select **Generate new token**.

Use the following minimum settings:

- Token name: `FM2Edge Colab read-only`
- Expiration: a short period appropriate for the PoC, such as 30 days
- Resource owner: `tiny-ticop`
- Repository access: **Only select repositories** > `FM2Edge`
- Repository permissions: **Contents: Read-only**

Metadata read access is included automatically. No write, administration,
workflow, package, or organization permission is needed for Colab training.
Copy the token when GitHub shows it; GitHub does not display it again.

## 2. Add the token to Colab Secrets

1. While signed in to GitHub, open
   `notebooks/phase1_5_oxford_pet_colab.ipynb` and download the raw file.
2. Open Colab and select **File > Upload notebook** to upload that file. Colab's
   GitHub tab may also be used after authorizing access to private repositories.
3. Open the key-shaped **Secrets** panel in the left sidebar.
4. Add a secret named exactly `GITHUB_TOKEN`.
5. Paste the token as its value.
6. Enable notebook access for that secret.

Do not paste the token into a code cell, notebook text, Git remote URL, Drive,
issue, or chat message.

## 3. Run and revoke

Select a GPU runtime and run the notebook from the first cell. The repository
cell clones `https://github.com/tiny-ticop/FM2Edge.git`; later runs perform a
fast-forward-only pull.

After the PoC, revoke the token in GitHub or let its short expiration end. If a
token is ever exposed, revoke it immediately and create a replacement.

Official reference: [GitHub personal access token
guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
