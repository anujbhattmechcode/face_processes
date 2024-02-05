from API import facial_process_api

if __name__ == '__main__':
    facial_process_api.run(
        '0.0.0.0',
        5010,
        debug=True
    )
