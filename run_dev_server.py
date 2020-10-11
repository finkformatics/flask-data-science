from flask_app import app


def main() -> None:
    app.run(host='localhost', port=9999)


if __name__ == '__main__':
    main()
