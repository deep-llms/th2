"""Dev-only Dropbox shared-folder listing and file downloads; no uploads.

No credentials/shared URLs are printed. Downloads are atomic and never
overwrite existing files. This script does not interact with GPU processes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class AccessError(RuntimeError):
    pass


def read_fields(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in result:
                raise AccessError("Duplicate field in local credentials/folders file")
            result[key] = value.strip()
    return result


def shared_link(path, label):
    url = read_fields(path).get(label)
    if not url:
        raise AccessError("Requested folder label is missing")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in {"dropbox.com", "www.dropbox.com"} or parts.username or parts.password:
        raise AccessError("Unexpected shared-link host")
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                      urlencode([(k, v) for k, v in parse_qsl(parts.query) if k != "dl"]), ""))


def request(url, body, headers):
    try:
        return urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=120)
    except HTTPError as error:
        # Response bodies and request headers may contain sensitive metadata.
        raise AccessError(f"Dropbox HTTP {error.code}; remote status is unverified") from None
    except (URLError, TimeoutError):
        raise AccessError("Dropbox connection failed; remote status is unverified") from None


def access_token(credentials_path):
    if os.environ.get("DROPBOX_ACCESS_TOKEN"):
        return os.environ["DROPBOX_ACCESS_TOKEN"]
    credentials = read_fields(credentials_path)
    if any(not credentials.get(k) for k in ("app_key", "app_secret", "refresh_token")):
        raise AccessError("Missing credential fields: need app_key, app_secret, refresh_token")
    body = urlencode({"grant_type": "refresh_token",
                      "client_id": credentials["app_key"],
                      "client_secret": credentials["app_secret"],
                      "refresh_token": credentials["refresh_token"]}).encode()
    with request("https://api.dropboxapi.com/oauth2/token", body,
                 {"Content-Type": "application/x-www-form-urlencoded"}) as response:
        data = json.load(response)
    if not isinstance(data.get("access_token"), str) or not data["access_token"]:
        raise AccessError("Token refresh returned no access token")
    return data["access_token"]


def validate_path(path):
    if path and (not path.startswith("/") or ".." in Path(path).parts):
        raise AccessError("Use a path inside the share beginning with /; no parent traversal")


def list_folder(token, url, path=""):
    validate_path(path)
    endpoint = "files/list_folder"
    body = {"path": path, "shared_link": {"url": url}, "recursive": False, "limit": 2000}
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    seen_cursors = set()
    while True:
        with request("https://api.dropboxapi.com/2/" + endpoint,
                     json.dumps(body).encode(), headers) as response:
            data = json.load(response)
        for item in data["entries"]:
            yield {"type": item[".tag"], "path": path.rstrip("/") + "/" + item["name"],
                   "bytes": item.get("size"), "modified": item.get("server_modified")}
        if not data.get("has_more"):
            break
        cursor = data["cursor"]
        if cursor in seen_cursors:
            raise AccessError("Dropbox repeated a pagination cursor")
        seen_cursors.add(cursor)
        endpoint = "files/list_folder/continue"
        body = {"cursor": cursor}


def download(token, url, remote_path, destination, expected_sha256=None):
    validate_path(remote_path)
    if not remote_path or remote_path.endswith("/"):
        raise AccessError("Download requires an exact file path, not a folder")
    if expected_sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise AccessError("Expected SHA256 must be 64 hexadecimal characters")
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise AccessError("Destination already exists; use a fresh filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": "Bearer " + token,
               "Dropbox-API-Arg": json.dumps({"url": url, "path": remote_path})}
    partial = None
    try:
        with request("https://content.dropboxapi.com/2/sharing/get_shared_link_file",
                     # No request body: b"" makes urllib add a form Content-Type,
                     # which is inappropriate for a content-download endpoint.
                     None, headers) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".dropbox-",
                                             suffix=".part", delete=False) as output:
                partial = Path(output.name)
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            expected_size = response.headers.get("Content-Length")
            if expected_size is not None and size != int(expected_size):
                raise AccessError("Incomplete download length")
        checksum = digest.hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256.lower():
            raise AccessError("Download SHA256 mismatch; no final file published")
        # Atomic no-clobber publication, including races with another downloader.
        os.link(partial, destination)
        return {"bytes": size, "sha256": checksum, "local_path": str(destination)}
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=Path("temp/dropbox_credentials.txt"))
    parser.add_argument("--folders", type=Path, default=Path("temp/dropbox_folders.txt"))
    parser.add_argument("--label", default="runner")
    sub = parser.add_subparsers(dest="mode", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--path", default="")
    pulling = sub.add_parser("download")
    pulling.add_argument("--path", required=True)
    pulling.add_argument("--output", required=True, type=Path)
    pulling.add_argument("--sha256")
    args = parser.parse_args()
    try:
        url = shared_link(args.folders, args.label)
        token = access_token(args.credentials)
        if args.mode == "list":
            for item in list_folder(token, url, args.path):
                print(json.dumps(item))
        else:
            print(json.dumps(download(token, url, args.path, args.output, args.sha256)))
    except (AccessError, OSError, ValueError, KeyError) as error:
        # Do not print exception bodies from credential parsing or API objects.
        message = str(error) if isinstance(error, AccessError) else type(error).__name__
        parser.exit(1, f"Dropbox retrieval failed: {message}\n")


if __name__ == "__main__":
    main()
