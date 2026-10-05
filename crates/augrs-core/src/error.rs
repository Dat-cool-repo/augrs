use std::fmt;

/// Errors raised by augrs.
#[derive(Debug, Clone, PartialEq)]
pub enum AugError {
    /// A transform was configured with invalid parameters.
    InvalidParam(String),
    /// The data passed in (image, mask, boxes, ...) is not valid for the pipeline.
    InvalidInput(String),
}

impl fmt::Display for AugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AugError::InvalidParam(s) => write!(f, "invalid parameter: {s}"),
            AugError::InvalidInput(s) => write!(f, "invalid input: {s}"),
        }
    }
}

impl std::error::Error for AugError {}

pub type Result<T> = std::result::Result<T, AugError>;

pub(crate) fn param<T>(msg: impl Into<String>) -> Result<T> {
    Err(AugError::InvalidParam(msg.into()))
}

pub(crate) fn input<T>(msg: impl Into<String>) -> Result<T> {
    Err(AugError::InvalidInput(msg.into()))
}
