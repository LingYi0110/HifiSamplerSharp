namespace HifiSampler.Core.Utils;

/// <summary>
/// Akima 1D interpolator, similar to scipy.interpolate.Akima1DInterpolator.
/// Uses Akima's method for smooth interpolation with reduced oscillation.
/// </summary>
public class AkimaInterpolator
{
    private readonly double[] _x;
    private readonly double[] _y;
    private readonly double[] _b; // first derivative coefficients
    private readonly double[] _c; // second derivative coefficients
    private readonly double[] _d; // third derivative coefficients

    /// <summary>
    /// Initializes a new instance of the Akima interpolator.
    /// </summary>
    /// <param name="x">Array of x values (must be strictly increasing)</param>
    /// <param name="y">Array of y values (same length as x)</param>
    /// <exception cref="ArgumentException">Thrown when input arrays are invalid</exception>
    public AkimaInterpolator(double[] x, double[] y)
    {
        if (x == null || y == null)
            throw new ArgumentNullException(x == null ? nameof(x) : nameof(y));
        
        if (x.Length != y.Length)
            throw new ArgumentException("Arrays x and y must have the same length.");
        
        if (x.Length < 2)
            throw new ArgumentException("At least 2 data points are required.");

        // Check that x is strictly increasing
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] <= x[i - 1])
                throw new ArgumentException("Array x must be strictly increasing.");
        }

        _x = (double[])x.Clone();
        _y = (double[])y.Clone();

        int n = x.Length;
        _b = new double[n - 1];
        _c = new double[n - 1];
        _d = new double[n - 1];

        ComputeCoefficients();
    }

    /// <summary>
    /// Initializes a new instance of the Akima interpolator using float arrays.
    /// </summary>
    public AkimaInterpolator(float[] x, float[] y)
        : this(x.Select(v => (double)v).ToArray(), y.Select(v => (double)v).ToArray())
    {
    }

    private void ComputeCoefficients()
    {
        int n = _x.Length;

        // Compute slopes between consecutive points
        double[] m = new double[n - 1];
        for (int i = 0; i < n - 1; i++)
        {
            m[i] = (_y[i + 1] - _y[i]) / (_x[i + 1] - _x[i]);
        }

        // For Akima interpolation, we need slopes at each point
        // We extend the slopes array by 2 on each side using the formula from Akima's paper
        double[] mExt = new double[n + 3];

        // Copy original slopes to the middle
        for (int i = 0; i < n - 1; i++)
        {
            mExt[i + 2] = m[i];
        }

        // Extend slopes at the beginning (using Akima's extrapolation)
        if (n >= 3)
        {
            mExt[1] = 2 * m[0] - m[1];
            mExt[0] = 2 * mExt[1] - m[0];
        }
        else
        {
            mExt[1] = m[0];
            mExt[0] = m[0];
        }

        // Extend slopes at the end
        if (n >= 3)
        {
            mExt[n + 1] = 2 * m[n - 2] - m[n - 3];
            mExt[n + 2] = 2 * mExt[n + 1] - m[n - 2];
        }
        else
        {
            mExt[n + 1] = m[n - 2];
            mExt[n + 2] = m[n - 2];
        }

        // Compute the derivative at each point using Akima's formula
        double[] t = new double[n];
        for (int i = 0; i < n; i++)
        {
            double m1 = mExt[i];
            double m2 = mExt[i + 1];
            double m3 = mExt[i + 2];
            double m4 = mExt[i + 3];

            double w1 = Math.Abs(m4 - m3);
            double w2 = Math.Abs(m2 - m1);

            if (w1 + w2 < 1e-15)
            {
                // If weights are essentially zero, use simple average
                t[i] = (m2 + m3) / 2.0;
            }
            else
            {
                t[i] = (w1 * m2 + w2 * m3) / (w1 + w2);
            }
        }

        // Compute cubic polynomial coefficients for each interval
        for (int i = 0; i < n - 1; i++)
        {
            double dx = _x[i + 1] - _x[i];
            double dy = _y[i + 1] - _y[i];

            _b[i] = t[i];
            _c[i] = (3 * dy / dx - 2 * t[i] - t[i + 1]) / dx;
            _d[i] = (t[i] + t[i + 1] - 2 * dy / dx) / (dx * dx);
        }
    }

    /// <summary>
    /// Interpolates the value at the given x coordinate.
    /// </summary>
    /// <param name="xi">The x coordinate to interpolate at</param>
    /// <returns>The interpolated y value</returns>
    public double Interpolate(double xi)
    {
        int n = _x.Length;

        // Handle boundary cases
        if (xi <= _x[0])
        {
            // Extrapolate using the first segment
            double dx = xi - _x[0];
            return _y[0] + _b[0] * dx + _c[0] * dx * dx + _d[0] * dx * dx * dx;
        }

        if (xi >= _x[n - 1])
        {
            // Extrapolate using the last segment
            int last = n - 2;
            double dx = xi - _x[last];
            return _y[last] + _b[last] * dx + _c[last] * dx * dx + _d[last] * dx * dx * dx;
        }

        // Binary search to find the interval
        int idx = BinarySearch(xi);
        double dxi = xi - _x[idx];

        return _y[idx] + _b[idx] * dxi + _c[idx] * dxi * dxi + _d[idx] * dxi * dxi * dxi;
    }

    /// <summary>
    /// Interpolates values at multiple x coordinates.
    /// </summary>
    /// <param name="xi">Array of x coordinates to interpolate at</param>
    /// <returns>Array of interpolated y values</returns>
    public double[] Interpolate(double[] xi)
    {
        double[] result = new double[xi.Length];
        for (int i = 0; i < xi.Length; i++)
        {
            result[i] = Interpolate(xi[i]);
        }
        return result;
    }

    /// <summary>
    /// Interpolates values at multiple x coordinates (float version).
    /// </summary>
    public float[] Interpolate(float[] xi)
    {
        float[] result = new float[xi.Length];
        for (int i = 0; i < xi.Length; i++)
        {
            result[i] = (float)Interpolate(xi[i]);
        }
        return result;
    }

    /// <summary>
    /// Interpolates values at x coordinates specified by a span.
    /// </summary>
    public void Interpolate(ReadOnlySpan<double> xi, Span<double> result)
    {
        if (xi.Length != result.Length)
            throw new ArgumentException("Input and output spans must have the same length.");

        for (int i = 0; i < xi.Length; i++)
        {
            result[i] = Interpolate(xi[i]);
        }
    }

    /// <summary>
    /// Binary search to find the interval index for a given x value.
    /// Returns the index i such that x[i] <= xi < x[i+1]
    /// </summary>
    private int BinarySearch(double xi)
    {
        int lo = 0;
        int hi = _x.Length - 2;

        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (_x[mid] <= xi)
                lo = mid;
            else
                hi = mid - 1;
        }

        return lo;
    }

    /// <summary>
    /// Gets the x values used for interpolation.
    /// </summary>
    public ReadOnlySpan<double> XValues => _x;

    /// <summary>
    /// Gets the y values used for interpolation.
    /// </summary>
    public ReadOnlySpan<double> YValues => _y;
}