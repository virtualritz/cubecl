use super::logger::{LogLevel, LoggerConfig};

/// Configuration for compilation settings in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Logger configuration for compilation logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
    /// Whether compiled kernels are cached in the active environment.
    #[serde(default)]
    #[cfg(persistence)]
    pub cache: bool,
    /// Controls whether kernel launches enforce bounds checks.
    #[serde(default)]
    pub check_mode: BoundsCheckMode,
    /// How far the CPU runtime carries an f16 intermediate in f32 before rounding it. `None`
    /// chooses by whether the host computes in f16 directly. Other runtimes ignore it.
    #[serde(default)]
    pub f16_evaluation: Option<F16Evaluation>,
    /// Whether compilers fuse a multiply and the add or subtract that consumes it into one
    /// `fma`. `None` is [`FpContraction::Fast`], and leaves the choice open to a library that
    /// sets the configuration with [`RuntimeConfig::try_set`](super::RuntimeConfig::try_set).
    #[serde(default)]
    pub fp_contraction: Option<FpContraction>,
}

/// Whether compilers fuse `a * b + c`, `a * b - c` and `c - a * b` into a fused multiply-add,
/// named after clang's `-ffp-contract`.
///
/// A fused multiply-add rounds once where the separate operations round twice, so a fused
/// result can differ from the same expression evaluated on the host. Code that compares device
/// output with a host reference can turn fusion off.
///
/// The setting controls the fusion pass of `CubeCL`'s own compilers. A backend compiler can
/// still fuse on its own: LLVM's contraction flags, `nvcc`'s `-fmad`, or a GPU driver.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FpContraction {
    /// Fuse a product into the add or subtract that consumes it, if nothing else uses it.
    #[default]
    #[serde(rename = "fast")]
    Fast,
    /// Keep every multiply and add separately rounded, as written.
    #[serde(rename = "off")]
    Off,
}

impl FpContraction {
    /// Whether compilers may fuse.
    pub fn allows_fusion(self) -> bool {
        matches!(self, Self::Fast)
    }
}

/// The [`FpContraction`] of the global [`CubeClRuntimeConfig`](super::CubeClRuntimeConfig).
pub fn fp_contraction() -> FpContraction {
    use super::RuntimeConfig as _;
    super::CubeClRuntimeConfig::get()
        .compilation
        .fp_contraction
        .unwrap_or_default()
}

/// How far an f32 intermediate is allowed to travel before it is rounded back to f16.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum F16Evaluation {
    /// Round after every operation, which is what a GPU does. Where f16 is not native it costs a
    /// convert pair per operation.
    #[serde(rename = "per-operation")]
    PerOperation,
    /// Round where a value is stored, a `let mut` included, or read by anything but arithmetic.
    /// The default where f16 is not native.
    #[default]
    #[serde(rename = "chain")]
    Chain,
    /// Also hold a private f16 variable in f32 where that removes more converts than it adds, so
    /// a running total read often enough inside its loop stays in f16. Costs vector registers, so
    /// a wide kernel may want a narrower line.
    #[serde(rename = "accumulators")]
    Accumulators,
}

impl F16Evaluation {
    /// The mode for a host that does or does not compute in f16 directly. Where it does, a chain
    /// held in f32 only adds converts.
    pub fn for_native_f16(native: bool) -> Self {
        match native {
            true => Self::PerOperation,
            false => Self::Chain,
        }
    }
}

impl core::fmt::Display for F16Evaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::PerOperation => "per-operation",
            Self::Chain => "chain",
            Self::Accumulators => "accumulators",
        })
    }
}

/// Bounds checks options.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum BoundsCheckMode {
    #[serde(rename = "enforce")]
    /// Always enforce bounds checks on every kernel launch.
    Enforce,
    #[serde(rename = "validate")]
    /// Always enforce bounds checks on every kernel launch, and validate unchecked kernels for OOB.
    Validate,
    /// Enforce bounds checking on standard launches, but skip checks on
    /// explicitly unchecked launches for better performance.
    #[default]
    #[serde(rename = "auto")]
    Auto,
}

/// Log levels for compilation in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompilationLogLevel {
    /// Compilation logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic compilation information is logged such as when kernels are compiled.
    #[serde(rename = "basic")]
    Basic,

    /// Full compilation details are logged including source code.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for CompilationLogLevel {}
