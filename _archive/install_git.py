import urllib.request
import subprocess
import os

portable_exe = os.path.expanduser(r"~\Downloads\PortableGit.exe")
git_install_dir = r"C:\Users\user\AppData\Local\Programs\Git"
installer_exe = os.path.expanduser(r"~\Downloads\Git-Installer-64bit.exe")

print("[*] 1/3: Downloading PortableGit (Full Git + Git Credential Manager + Git Bash)...")
p_url = "https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.4/PortableGit-2.56.0.4-64-bit.7z.exe"
urllib.request.urlretrieve(p_url, portable_exe)
print(f"[+] Downloaded {os.path.getsize(portable_exe)} bytes.")

print(f"[*] 2/3: Extracting to {git_install_dir}...")
cmd = f'"{portable_exe}" -y -o"{git_install_dir}"'
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(f"[+] Extraction complete (code {res.returncode})")

# Clean up temporary extractor
if os.path.exists(portable_exe):
    os.remove(portable_exe)

print(f"[*] 3/3: Downloading the official Git Windows Setup wizard to {installer_exe}...")
inst_url = "https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.4/Git-2.56.0.4-64-bit.exe"
urllib.request.urlretrieve(inst_url, installer_exe)
print(f"[+] Downloaded setup wizard installer ({os.path.getsize(installer_exe)} bytes).")

git_bin = os.path.join(git_install_dir, "cmd", "git.exe")
print(f"[+] Git verified at: {git_bin} (Exists: {os.path.exists(git_bin)})")

