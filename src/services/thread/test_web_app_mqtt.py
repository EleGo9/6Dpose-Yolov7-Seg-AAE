import sys
import threading
import argparse

sys.path.append("src/config")
from web_app_configuration import WebAppConfiguration
from database_configuration import DatabaseConfiguration

sys.path.append("src/services/webapp")
from web_app import WebApp

sys.path.append("src/services/thread/webapp")
from web_app_thread import WebAppThread


def pipeline():
    web_app = WebApp(WebAppConfiguration.IP_ADDRESS, WebAppConfiguration.PORT, WebAppConfiguration.DEBUG, DatabaseConfiguration.PATH)
    web_app_thread = WebAppThread(web_app)

    web_app_threading = threading.Thread(target=web_app_thread.run)

    web_app_threading.start()

    web_app_threading.join()


def main(args):
    try:
        if args.cycle == "pipeline":
            pipeline()
        else:
            print("Source images not correct")
            exit(-1)

    except Exception as e:
        sys.stderr.write(str(e) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", required=False, help="pipeline, pnp, methods", default="pipeline")
    args = parser.parse_args()
    main(args)