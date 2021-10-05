# Use Docker

The Dockerfile is set up to expose port 22 and make the container accessible via SSH. This allows not only to
transfer files, but also to use the Docker container as a remote debugger for PyCharm and other development environments
. However, for this to work, the public SSH key must be included in the container.

1. Generate a SSH-Key:
    ```bash
    ssh-keygen -t rsa
    ```

2. Copy the created public key in a folder called `ssh_keys`

3. Create an archive which contains the public key:
   ```bash
   tar -czvf ssh_keys.tar.gz /path/to/ssh_keys
   ```
   
4. Save the `Dockerfile`, the archive `ssh_keys.tar.gz` and the file `requirements.txt` in the same Folder and run in
 this folder the command
   ```bash
   docker build -t fwa .
   ```