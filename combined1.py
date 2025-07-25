#!/usr/bin/env python3
"""
Combined Streamlit Dashboard (Tasks + Linux Executor + Docker + File Manager)
============================================================================
Four distinct workspaces in one page:

1. *Tasks Dashboard* – WhatsApp message scheduler, email sender, Twilio voice/SMS,
   RAM stats, Google top‑5 results, face‑swap demo (OpenCV), random art,
   simple web scraper.
2. *Linux Executor* – Red Hat command cheatsheet and generic SSH executor.
3. *Docker Menu (SSH)* – 50+ curated one‑click Docker commands *plus* a typo‑tolerant
   free‑form box, all executed on a remote host via password‑only SSH (port 22).
4. *Secure File Manager* – Browse, upload, download, rename, delete files and folders
   with file type visualization.

Quick start::
    pip install streamlit pywhatkit googlesearch-python psutil twilio numpy \
                opencv-python beautifulsoup4 requests paramiko matplotlib
    streamlit run combined_dashboard.py
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import shlex
import difflib
import shutil
from typing import Dict, Tuple
from pathlib import Path
from collections import Counter

import streamlit as st
import paramiko
import matplotlib.pyplot as plt

# "Tasks" libs
import pywhatkit
from googlesearch import search
import psutil
from twilio.rest import Client
import numpy as np
import cv2
from bs4 import BeautifulSoup
import requests

# ---------------------------------------------------------------------------
# Credentials & demo users  (replace with env vars / DB in production)
# ---------------------------------------------------------------------------
TWILIO_SID   = "AC2ccec04e77d14d8b43bad3e0e07a6598"
TWILIO_TOKEN = "7e1bde6382bc5d07e328f0bb9f83345c"
TWILIO_NUMBER = "‪+14173522775‬"
#FILE_MANAGER_PASSWORD = "admin123"  # Change this to your desired password

#USER_CREDENTIALS = {"admin": "admin123", "udit": "abcd"}

# ---------------------------------------------------------------------------
# Docker command catalog  (label → (shell template, needs_arg?))
# ---------------------------------------------------------------------------
COMMANDS: Dict[str, Tuple[str, bool]] = {
    # Basics
    "Docker Version": ("docker --version", False),
    "Docker Info": ("docker info", False),
    "List Images": ("docker images", False),
    "List Containers (all)": ("docker ps -a", False),
    "Run hello-world": ("docker run --rm hello-world", False),

    # Image / container mgmt
    "Pull Image (name)": ("docker pull {arg}", True),
    "Remove Image (name/id)": ("docker rmi {arg}", True),
    "Create Container (name) from alpine": ("docker create --name {arg} alpine", True),
    "Start Container": ("docker start {arg}", True),
    "Stop Container": ("docker stop {arg}", True),
    "Remove Container": ("docker rm {arg}", True),
    "Container Logs": ("docker logs {arg}", True),
    "Exec Shell (/bin/sh)": ("docker exec -it {arg} /bin/sh", True),
    "Live Stats": ("docker stats --no-stream", False),

    # Cleanup
    "System Prune (all)": ("docker system prune -f", False),
    "Prune Dangling Images": ("docker image prune -f", False),
    "Prune Volumes": ("docker volume prune -f", False),

    # Networks & volumes
    "List Networks": ("docker network ls", False),
    "Create Network": ("docker network create {arg}", True),
    "Remove Network": ("docker network rm {arg}", True),
    "List Volumes": ("docker volume ls", False),
    "Create Volume": ("docker volume create {arg}", True),
    "Remove Volume": ("docker volume rm {arg}", True),

    # Tag & push
    "Tag Image": ("docker tag {arg}", True),
    "Push Image": ("docker push {arg}", True),

    # Inspect / copy
    "Inspect Container": ("docker inspect {arg}", True),
    "Inspect Image": ("docker inspect {arg}", True),
    "Copy out (ctr:path dest)": ("docker cp {arg}", True),
    "Disk Usage": ("docker system df", False),
    "Image History": ("docker history {arg}", True),

    # Registry & context
    "Login to Registry": ("docker login", False),
    "Logout from Registry": ("docker logout", False),
    "List Contexts": ("docker context ls", False),
    "Switch Context": ("docker context use {arg}", True),

    # Compose
    "Compose Version": ("docker compose version", False),
    "Compose Up (detached)": ("docker compose up -d", False),
    "Compose Down": ("docker compose down", False),
    "Compose Logs": ("docker compose logs --tail 50", False),

    # Builder / save / load
    "List Builder Cache": ("docker builder ls", False),
    "Prune Builder Cache": ("docker builder prune -f", False),
    "Builder Build (Dockerfile)": ("docker build -t {arg}", True),
    "Save Image → tar": ("docker save {arg}", True),
    "Load Image from tar": ("docker load -i {arg}", True),

    # Advanced ops
    "Top (processes in ctr)": ("docker top {arg}", True),
    "Checkpoint create": ("docker checkpoint create {arg}", True),
    "Checkpoint list": ("docker checkpoint ls {arg}", True),
    "Checkpoint rm": ("docker checkpoint rm {arg}", True),
    "Image Digests": ("docker image ls --digests", False),
    "Events (10s)": ("timeout 10 docker events", False),
    "Rename Container": ("docker rename {arg}", True),
    "Commit Container → Image": ("docker commit {arg}", True),
    "Update Container Resources": ("docker update {arg}", True),

    # Exit sentinel
    "Exit": ("exit", False),
}

SUBCOMMANDS = {
    "attach","build","builder","checkpoint","commit","compose","config","container","context",
    "cp","create","diff","events","exec","export","history","image","images","import","info",
    "inspect","kill","load","login","logout","logs","network","pause","port","ps","pull","push",
    "rename","restart","rm","rmi","run","save","scan","search","secret","service","stack",
    "start","stats","stop","swarm","system","tag","top","trust","unpause","update","version",
    "volume","wait",
}

# ---------------------------------------------------------------------------
# Helper – gentle auto-correction for free-form Docker commands
# ---------------------------------------------------------------------------
def autocorrect_cmd(cmd: str) -> tuple[str, str]:
    """Return (corrected_cmd, fix_note) – fix note is '' if nothing touched."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return cmd, "Could not parse command; running verbatim."

    if not tokens:
        return cmd, "Empty command; nothing to run."

    note = ""

    # Prefix 'docker' if typo or missing
    if tokens[0] != "docker":
        if difflib.SequenceMatcher(None, tokens[0], "docker").ratio() > 0.6:
            note += f"Auto-corrected '{tokens[0]}' → 'docker'.  "
            tokens[0] = "docker"
        elif tokens[0] in SUBCOMMANDS:
            note += "Inserted missing 'docker' prefix.  "
            tokens.insert(0, "docker")

    # Fix sub-command typos
    if len(tokens) >= 2 and tokens[0] == "docker":
        sub = tokens[1]
        if sub not in SUBCOMMANDS:
            close = difflib.get_close_matches(sub, SUBCOMMANDS, n=1)
            if close and difflib.SequenceMatcher(None, sub, close[0]).ratio() > 0.6:
                note += f"Auto-corrected sub-command '{sub}' → '{close[0]}'.  "
                tokens[1] = close[0]

    return " ".join(tokens), note.strip()

# ---------------------------------------------------------------------------
# Streamlit page config & session defaults
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Python Tasks + Linux + Docker + File Manager Dashboard", 
    page_icon="🔧", 
    layout="wide"
)

for key, default in (
    ("logged_in", False),
    ("username", ""),
    ("docker_client", None),
    ("linux_client", None),
    ("file_manager_authenticated", False),
):
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------------------
# Authentication views
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sidebar – choose workspace
# ---------------------------------------------------------------------------
#st.sidebar.success(f"Logged in as *{st.session_state.username}*")
#st.sidebar.button("Logout", on_click=logout)

workspace = st.sidebar.radio(
    "Workspace", 
    [
        "Tasks Dashboard", 
        "Linux Executor", 
        "Docker Menu (SSH)",
        "Secure File Manager"
    ]
)

# ===========================================================================
#                                TASKS DASHBOARD
# ===========================================================================
if workspace == "Tasks Dashboard":
    st.title("🛠 Tasks Dashboard")

    task = st.sidebar.selectbox(
        "Choose a Task",
        [
            "WhatsApp Automation",
            "Email Sender",
            "Twilio Call",
            "Send SMS",
            "System RAM Info",
            "Google Search",
            "Face Swap via OpenCV",
            "Random Art",
            "Web Scraper",
        ],
    )

    # --------------------------- WhatsApp ----------------------------------
    if task == "WhatsApp Automation":
        st.header("📲 Send WhatsApp Message")
        phone = st.text_input("Recipient Phone Number", "+91")
        message = st.text_input("Message", "Hello")
        hour = st.number_input("Hour (24H)", 0, 23, 12)
        minute = st.number_input("Minute", 0, 59, 0)
        if st.button("Send Message"):
            pywhatkit.sendwhatmsg(phone, message, int(hour), int(minute))
            st.success("Message scheduled!")

    # --------------------------- Email -------------------------------------
    elif task == "Email Sender":
        st.header("📧 Send Email")
        sender = st.text_input("G-Mail Address")
        app_pass = st.text_input("App Password", type="password")
        subj = st.text_input("Subject")
        body = st.text_area("Email Body")
        recipient = st.text_input("Recipient Email")
        if st.button("Send Email"):
            pywhatkit.send_mail(sender, app_pass, subj, body, recipient)
            st.success("Email sent!")

    # --------------------------- Twilio Call -------------------------------
    elif task == "Twilio Call":
        st.header("📞 Place a Call (Twilio)")
        to_num = st.text_input("Recipient Number")
        if st.button("Call Now"):
            try:
                client = Client(TWILIO_SID, TWILIO_TOKEN)
                call = client.calls.create(
                    url="http://demo.twilio.com/docs/classic.mp3",
                    from_=TWILIO_NUMBER,
                    to=to_num,
                )
                st.success(f"Call placed! SID {call.sid}")
            except Exception as e:
                st.error(f"Error: {e}")

    # --------------------------- SMS ---------------------------------------
    elif task == "Send SMS":
        st.header("✉ Send SMS (Twilio)")
        to_num = st.text_input("Recipient Number")
        msg = st.text_input("Message Body")
        if st.button("Send SMS"):
            try:
                client = Client(TWILIO_SID, TWILIO_TOKEN)
                res = client.messages.create(body=msg, from_=TWILIO_NUMBER, to=to_num)
                st.success(f"SMS queued. SID {res.sid}")
            except Exception as e:
                st.error(f"Error: {e}")

    # --------------------------- RAM Stats ---------------------------------
    elif task == "System RAM Info":
        st.header("💻 System Memory")
        mem = psutil.virtual_memory()
        st.write(f"*Total:* {mem.total/1e9:.2f} GB")
        st.write(f"*Available:* {mem.available/1e9:.2f} GB")
        st.write(f"*Used:* {mem.used/1e9:.2f} GB ({mem.percent} %)")

    # --------------------------- Google Search -----------------------------
    elif task == "Google Search":
        st.header("🔎 Google Top-5")
        q = st.text_input("Search query")
        if st.button("Search"):
            try:
                hits = list(search(q, num_results=5))
                st.text_area("Results", "\n".join(f"{i+1}. {u}" for i, u in enumerate(hits)), height=160)
            except Exception as e:
                st.error(f"Error: {e}")

    # --------------------------- Face Swap ---------------------------------
    elif task == "Face Swap via OpenCV":
        st.header("😎 Face Swap (webcam)")
        st.warning("Grabs your webcam – press *SPACE* twice to snap two faces.")
        if st.button("Start"):
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            imgs = []
            while len(imgs) < 2:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imshow("Capture faces (ESC to abort)", frame)
                k = cv2.waitKey(1)
                if k == 32:  # SPACE
                    imgs.append(frame.copy())
                    print("Captured", len(imgs))
                elif k == 27:  # ESC
                    break
            cap.release()
            cv2.destroyAllWindows()

            def crop(face_img):
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                return (
                    (faces[0], face_img[faces[0][1] : faces[0][1] + faces[0][3], faces[0][0] : faces[0][0] + faces[0][2]])
                    if len(faces)
                    else (None, None)
                )

            if len(imgs) == 2:
                (box1, f1), (box2, f2) = crop(imgs[0]), crop(imgs[1])
                if f1 is not None and f2 is not None:
                    x, y, w, h = box1
                    f2r = cv2.resize(f2, (w, h))
                    imgs[0][y : y + h, x : x + w] = f2r
                    cv2.imshow("Swapped", imgs[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    st.success("Done!")
                else:
                    st.error("Face detection failed.")
            else:
                st.warning("Need two snapshots.")

    # --------------------------- Random Art --------------------------------
    elif task == "Random Art":
        st.header("🎨 Random Circles")
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        for _ in range(100):
            c = tuple(np.random.randint(0, 500, 2))
            r = int(np.random.randint(10, 50))
            col = tuple(int(x) for x in np.random.randint(0, 255, 3))
            cv2.circle(img, c, r, col, -1)
        st.image(img[:, :, ::-1], caption="Random circles")

    # --------------------------- Web Scraper -------------------------------
    elif task == "Web Scraper":
        st.header("🌐 Simple Web Scraper")
        url = st.text_input("URL", "https://example.com")
        if st.button("Scrape"):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, "html.parser")
                st.subheader("Title")
                st.write(soup.title.string if soup.title else "No title")
                st.subheader("Text")
                st.text_area("Body", soup.get_text("\n"), height=300)
            except Exception as e:
                st.error(f"Error: {e}")

# ===========================================================================
#                                LINUX EXECUTOR
# ===========================================================================
elif workspace == "Linux Executor":
    st.title("🐧 Linux Command Executor")
    
    # Sidebar connection for Linux
    st.sidebar.subheader("Linux SSH Connection")
    l_host = st.sidebar.text_input("Host", key="l_host")
    l_user = st.sidebar.text_input("Username", value="root", key="l_user")
    l_pass = st.sidebar.text_input("Password", type="password", key="l_pass")

    if st.sidebar.button("Connect / Reconnect"):
        if not (l_host and l_user and l_pass):
            st.sidebar.error("Host / user / password required")
        else:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(l_host, 22, l_user, l_pass, timeout=30, banner_timeout=30)
                st.session_state.linux_client = client
                st.sidebar.success("Connected ✔")
            except Exception as e:
                st.session_state.linux_client = None
                st.sidebar.error(f"Connect failed: {e}")

    if st.sidebar.button("Disconnect"):
        if st.session_state.linux_client:
            try:
                st.session_state.linux_client.close()
            except:
                pass
        st.session_state.linux_client = None
        st.sidebar.info("Disconnected")

    client: paramiko.SSHClient | None = st.session_state.linux_client

    # Tabs for Cheatsheet and Executor
    tab1, tab2 = st.tabs(["Red Hat Cheatsheet", "Command Executor"])

    with tab1:
        st.header("📘 Red Hat Command Cheatsheet")
        st.write("Common Red Hat Linux commands for reference:")
        
        categories = {
            "File Operations": [
                "ls", "pwd", "cd", "mkdir", "rmdir", "rm -r", "cp", "mv", "touch", "cat",
                "more", "less", "head", "tail", "find", "locate", "stat"
            ],
            "Permissions": [
                "chmod", "chown", "chgrp", "umask"
            ],
            "Package Management": [
                "yum install", "yum remove", "yum update", "rpm -ivh", "dnf install"
            ],
            "Process Management": [
                "ps aux", "top", "htop", "kill", "killall", "free -h"
            ],
            "Disk Management": [
                "df -h", "du -sh"
            ],
            "Networking": [
                "ip addr", "ping", "curl", "wget", "netstat -tuln", "ss -tuln", "scp", "ssh"
            ],
            "User Management": [
                "adduser", "passwd", "userdel", "groupadd", "usermod -aG"
            ],
            "System Management": [
                "reboot", "shutdown -h now", "systemctl status", "systemctl restart", "journalctl -xe"
            ]
        }

        for category, commands in categories.items():
            with st.expander(category):
                for cmd in commands:
                    st.code(cmd)

    with tab2:
        st.header("🔐 Run Linux Command over SSH")
        
        if client is None:
            st.warning("Please connect to a Linux host first using the sidebar")
        else:
            cmd = st.text_input("Command to execute", "ls -l")
            if st.button("Execute"):
                try:
                    stdin, stdout, stderr = client.exec_command(cmd)
                    out = stdout.read().decode()
                    err = stderr.read().decode()
                    exit_code = stdout.channel.recv_exit_status()
                    
                    st.text_area("Output", out or "(no output)", height=300)
                    if err:
                        st.error(f"Error:\n{err}")
                    if exit_code != 0:
                        st.error(f"Command exited with code {exit_code}")
                    else:
                        st.success("Command executed successfully")
                except Exception as e:
                    st.error(f"SSH error: {e}")

# ===========================================================================
#                                DOCKER MENU
# ===========================================================================
elif workspace == "Docker Menu (SSH)":
    st.title("🐳 Docker Menu over SSH")

    # --------------- Sidebar connection pane ------------------------------
    st.sidebar.subheader("Docker SSH Connection")
    d_host = st.sidebar.text_input("Host", key="d_host")
    d_user = st.sidebar.text_input("Username", value="root", key="d_user")
    d_pass = st.sidebar.text_input("Password", type="password", key="d_pass")

    if st.sidebar.button("Connect / Reconnect"):
        if not (d_host and d_user and d_pass):
            st.sidebar.error("Host / user / password required")
        else:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(d_host, 22, d_user, d_pass, timeout=30, banner_timeout=30)
                st.session_state.docker_client = client
                st.sidebar.success("Connected ✔")
            except Exception as e:
                st.session_state.docker_client = None
                st.sidebar.error(f"Connect failed: {e}")

    if st.sidebar.button("Disconnect"):
        if st.session_state.docker_client:
            try:
                st.session_state.docker_client.close()
            except:
                pass
        st.session_state.docker_client = None
        st.sidebar.info("Disconnected")

    client: paramiko.SSHClient | None = st.session_state.docker_client

    # ---------------- Command picker --------------------------------------
    choice = st.selectbox(
        "Pick a Docker command:",
        ["📝 Custom command"] + sorted(COMMANDS.keys()),
    )

    cmd_to_run, fix_note = "", ""
    if choice == "📝 Custom command":
        raw = st.text_area("Enter full command", "docker ps -a", height=70)
        if raw.strip():
            cmd_to_run, fix_note = autocorrect_cmd(raw.strip())
    else:
        tmpl, needs_arg = COMMANDS[choice]
        if needs_arg:
            arg = st.text_input("Required argument(s)")
            cmd_to_run = tmpl.format(arg=arg) if arg else ""
        else:
            cmd_to_run = tmpl

    if st.button("▶ Run"):
        if client is None:
            st.error("Connect first")
        elif not cmd_to_run:
            st.warning("No command specified")
        else:
            if fix_note:
                st.info(fix_note)
            st.markdown(f"*Running:* {cmd_to_run}")
            try:
                stdin, stdout, stderr = client.exec_command(cmd_to_run)
                output = stdout.read().decode() + stderr.read().decode()
                exit_code = stdout.channel.recv_exit_status()
                st.code(output or "(no output)")
                if exit_code == 0:
                    st.success("Done ✓")
                else:
                    st.error(f"Exit code {exit_code}")
            except Exception as e:
                st.error(f"SSH error: {e}")

    st.caption(
        "© 2025 Docker SSH Menu – auto-corrects minor typos and runs both one-click "
        "and free-form Docker commands on a remote host (password auth, port 22)."
    )

# ===========================================================================
#                                FILE MANAGER
# ===========================================================================
elif workspace == "Secure File Manager":
    st.title("🔒 Secure File Management Dashboard")

    # File Manager authentication (separate from main login)
    if not st.session_state.file_manager_authenticated:
        entered_pw = st.text_input("Enter File Manager password", type="password")
        if st.button("Login to File Manager"):
            if entered_pw == FILE_MANAGER_PASSWORD:
                st.session_state.file_manager_authenticated = True
                st.success("Access granted.")
                st.experimental_rerun()
            else:
                st.error("Incorrect password. Try again.")
        st.stop()

    # --- DIRECTORY SELECTION ---
    directory = st.text_input("📂 Enter the directory path:")

    if directory and os.path.exists(directory):
        st.markdown("### 📜 Files & Folders")
        search_query = st.text_input("🔍 Search")

        files = os.listdir(directory)
        files = sorted(files)
        filtered_files = [f for f in files if search_query.lower() in f.lower()]

        file_types = []

        if not filtered_files:
            st.info("No matching files.")
        else:
            for f in filtered_files:
                full_path = os.path.join(directory, f)
                file_type = "📁 Folder" if os.path.isdir(full_path) else f"📄 File ({Path(f).suffix})"
                size = os.path.getsize(full_path) / 1024  # KB
                st.write(f"{f}** — {file_type} — {size:.2f} KB")
                if os.path.isfile(full_path):
                    with open(full_path, "rb") as file:
                        st.download_button("⬇ Download", data=file, file_name=f)
                    file_types.append(Path(f).suffix)

        # --- FILE TYPE CHART ---
        st.markdown("### 📊 File Type Distribution")
        if file_types:
            type_counts = Counter(file_types)
            fig, ax = plt.subplots()
            ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No files to visualize.")

        st.markdown("---")

        # --- UPLOAD ---
        st.subheader("📤 Upload File")
        uploaded_file = st.file_uploader("Choose a file to upload")
        if uploaded_file:
            save_path = os.path.join(directory, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded '{uploaded_file.name}' successfully.")

        st.markdown("---")

        # --- RENAME ---
        st.subheader("✏ Rename File/Folder")
        old_name = st.text_input("Old name")
        new_name = st.text_input("New name")
        if st.button("Rename"):
            try:
                os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))
                st.success("Renamed successfully.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")

        # --- DELETE ---
        st.subheader("🗑 Delete File or Directory")
        delete_name = st.text_input("Name to delete")
        if st.button("Delete"):
            try:
                path = os.path.join(directory, delete_name)
                if os.path.isfile(path):
                    os.remove(path)
                    st.success("File deleted.")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    st.success("Directory deleted.")
                else:
                    st.warning("Not found.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")

        # --- CREATE FOLDER ---
        st.subheader("📦 Create New Folder")
        folder_name = st.text_input("New folder name")
        if st.button("Create Directory"):
            try:
                os.makedirs(os.path.join(directory, folder_name), exist_ok=True)
                st.success("Folder created.")
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        if directory:
            st.error("Invalid directory path.")