import os
import subprocess

def docker_build_and_run(
    dockerfile_path=".",
    image_name="flower_recognition-app",
    container_name="flower_recognition",
    main_script="main.py",
    test_script="tests/test_app.py",
):
    """
    Build Docker image and run it with automatic test execution.
    """
    print("Building Docker image...")
    subprocess.run([
        "docker", "build", "-t", image_name, dockerfile_path
    ], check=True)

    print("Running container with tests and app...")
    subprocess.run([
        "docker", "run", "-it",  # -d
        "--name", container_name,
        "-p", "8000:8000",
        image_name
    ], check=True)

if __name__ == "__main__":
    docker_build_and_run()