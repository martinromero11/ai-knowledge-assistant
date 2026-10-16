"""Application errors. Each maps to an HTTP status and a stable error code."""


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"


class EmbeddingsNotConfigured(AppError):
    status_code = 400
    code = "embeddings_not_configured"


class InvalidDocument(AppError):
    status_code = 422
    code = "invalid_document"
