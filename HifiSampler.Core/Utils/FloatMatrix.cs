using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace HifiSampler.Core.Utils;

public struct FloatMatrix
{
    private const long ElementwiseParallelThreshold = 1_048_576;

    private int _rows;
    private int _cols;
    private int _stride;
    private float[] _buffer;

    public int Rows => _rows;
    public int Cols => _cols;
    public int Stride => _stride;
    public bool IsCompact => _stride == _cols;

    internal float[] Buffer => _buffer;

    public FloatMatrix(int rows, int cols, int? stride = null)
    {
        ValidateShape(rows, cols);

        int s = stride ?? cols;
        if (s < cols)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), "stride must be >= cols.");
        }

        _rows = rows;
        _cols = cols;
        _stride = s;
        _buffer = new float[checked(rows * s)];
    }

    public FloatMatrix(float[,] source, int? stride = null)
        : this(
            source is null ? throw new ArgumentNullException(nameof(source)) : source.GetLength(0),
            source.GetLength(1),
            stride)
    {
        CopyFromArray(source);
    }

    public static FloatMatrix FromArray(float[,] source, int? stride = null) => new(source, stride);

    public static FloatMatrix FromFlat(int rows, int cols, float[] source, bool takeOwnership = false)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        ValidateShape(rows, cols);

        int expected = checked(rows * cols);
        if (source.Length < expected)
        {
            throw new ArgumentException($"source length must be >= {expected}.", nameof(source));
        }

        if (takeOwnership && expected == source.Length)
        {
            var matrix = new FloatMatrix(rows, cols);
            matrix._buffer = source;
            return matrix;
        }

        return FromFlat(rows, cols, source.AsSpan(0, expected));
    }

    public static FloatMatrix FromFlat(int rows, int cols, ReadOnlySpan<float> source, int? stride = null)
    {
        ValidateShape(rows, cols);

        int expected = checked(rows * cols);
        if (source.Length < expected)
        {
            throw new ArgumentException($"source length must be >= {expected}.", nameof(source));
        }

        var matrix = new FloatMatrix(rows, cols, stride);
        if (rows == 0 || cols == 0)
        {
            return matrix;
        }

        if (matrix._stride == matrix._cols)
        {
            source[..expected].CopyTo(matrix._buffer.AsSpan(0, expected));
        }
        else
        {
            for (int row = 0; row < rows; row++)
            {
                source.Slice(row * cols, cols).CopyTo(matrix.RowSpan(row));
            }
        }

        return matrix;
    }

    public float this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _buffer[row * _stride + col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => _buffer[row * _stride + col] = value;
    }

    public Span<float> RowSpan(int row)
    {
        if ((uint)row >= (uint)_rows)
        {
            throw new ArgumentOutOfRangeException(nameof(row));
        }

        return _buffer.AsSpan(row * _stride, _cols);
    }

    public Span<float> AsSpan()
    {
        EnsureCompactLayout();
        return _buffer.AsSpan(0, _rows * _cols);
    }

    public ReadOnlySpan<float> AsReadOnlySpan()
    {
        EnsureCompactLayout();
        return _buffer.AsSpan(0, _rows * _cols);
    }

    public void CopyTo(Span<float> destination)
    {
        var expected = checked(_rows * _cols);
        if (destination.Length < expected)
        {
            throw new ArgumentException($"destination length must be >= {expected}.", nameof(destination));
        }

        if (_rows == 0 || _cols == 0)
        {
            return;
        }

        if (IsCompact)
        {
            _buffer.AsSpan(0, expected).CopyTo(destination);
            return;
        }

        for (int row = 0; row < _rows; row++)
        {
            RowSpan(row).CopyTo(destination.Slice(row * _cols, _cols));
        }
    }

    public float[,] ToArray2D()
    {
        var result = new float[_rows, _cols];
        if (_rows == 0 || _cols == 0)
        {
            return result;
        }

        if (IsCompact)
        {
            _buffer.AsSpan(0, _rows * _cols).CopyTo(MemoryMarshal.CreateSpan(ref result[0, 0], _rows * _cols));
            return result;
        }

        for (int row = 0; row < _rows; row++)
        {
            RowSpan(row).CopyTo(MemoryMarshal.CreateSpan(ref result[row, 0], _cols));
        }

        return result;
    }

    public void Clear() => Array.Clear(_buffer, 0, _buffer.Length);

    public static FloatMatrix operator *(FloatMatrix a, FloatMatrix b) => Multiply(a, b);

    public static FloatMatrix Add(FloatMatrix left, FloatMatrix right, bool parallel = false)
    {
        ValidateSameShape(left, right);
        var destination = new FloatMatrix(left._rows, left._cols);
        AddInto(left, right, destination, parallel);
        return destination;
    }

    public static void AddInto(FloatMatrix left, FloatMatrix right, FloatMatrix destination, bool parallel = false)
    {
        ValidateSameShape(left, right);
        ValidateSameShape(left, destination);

        int rows = left._rows;
        bool doParallel = parallel && ShouldParallelize(rows, left._cols);
        if (doParallel)
        {
            Parallel.For(0, rows, row => AddRow(left.RowSpan(row), right.RowSpan(row), destination.RowSpan(row)));
        }
        else
        {
            for (int row = 0; row < rows; row++)
            {
                AddRow(left.RowSpan(row), right.RowSpan(row), destination.RowSpan(row));
            }
        }
    }

    public static FloatMatrix Subtract(FloatMatrix left, FloatMatrix right, bool parallel = false)
    {
        ValidateSameShape(left, right);
        var destination = new FloatMatrix(left._rows, left._cols);
        SubtractInto(left, right, destination, parallel);
        return destination;
    }

    public static void SubtractInto(FloatMatrix left, FloatMatrix right, FloatMatrix destination, bool parallel = false)
    {
        ValidateSameShape(left, right);
        ValidateSameShape(left, destination);

        int rows = left._rows;
        bool doParallel = parallel && ShouldParallelize(rows, left._cols);
        if (doParallel)
        {
            Parallel.For(0, rows, row => SubtractRow(left.RowSpan(row), right.RowSpan(row), destination.RowSpan(row)));
        }
        else
        {
            for (int row = 0; row < rows; row++)
            {
                SubtractRow(left.RowSpan(row), right.RowSpan(row), destination.RowSpan(row));
            }
        }
    }

    public static FloatMatrix Scale(FloatMatrix source, float factor, bool parallel = false)
    {
        var destination = new FloatMatrix(source._rows, source._cols);
        ScaleInto(source, factor, destination, parallel);
        return destination;
    }

    public static void ScaleInto(FloatMatrix source, float factor, FloatMatrix destination, bool parallel = false)
    {
        ValidateSameShape(source, destination);

        int rows = source._rows;
        bool doParallel = parallel && ShouldParallelize(rows, source._cols);
        if (doParallel)
        {
            Parallel.For(0, rows, row => ScaleRow(source.RowSpan(row), destination.RowSpan(row), factor));
        }
        else
        {
            for (int row = 0; row < rows; row++)
            {
                ScaleRow(source.RowSpan(row), destination.RowSpan(row), factor);
            }
        }
    }

    public void ScaleInPlace(float factor, bool parallel = false)
    {
        ScaleInto(this, factor, this, parallel);
    }

    public static FloatMatrix Multiply(FloatMatrix a, FloatMatrix b, bool parallel = false)
    {
        ValidateMulShape(a, b);
        var c = new FloatMatrix(a._rows, b._cols);
        MultiplyIntoCore(a, b, c, parallel, -1);
        return c;
    }

    public static FloatMatrix Multiply(FloatMatrix a, FloatMatrix b, ParallelOptions? options)
    {
        ValidateMulShape(a, b);
        var c = new FloatMatrix(a._rows, b._cols);

        bool parallel = options is not null && options.MaxDegreeOfParallelism != 1;
        int maxDegreeOfParallelism = options?.MaxDegreeOfParallelism ?? -1;
        MultiplyIntoCore(a, b, c, parallel, maxDegreeOfParallelism);
        return c;
    }

    public static void MultiplyInto(FloatMatrix a, FloatMatrix b, FloatMatrix c, bool parallel = false)
    {
        ValidateMulShape(a, b);
        ValidateMulDestination(a, b, c);
        MultiplyIntoCore(a, b, c, parallel, -1);
    }

    public static void MultiplyInto(FloatMatrix a, FloatMatrix b, FloatMatrix c, ParallelOptions? options)
    {
        ValidateMulShape(a, b);
        ValidateMulDestination(a, b, c);

        bool parallel = options is not null && options.MaxDegreeOfParallelism != 1;
        int maxDegreeOfParallelism = options?.MaxDegreeOfParallelism ?? -1;
        MultiplyIntoCore(a, b, c, parallel, maxDegreeOfParallelism);
    }

    public void MultiplyInPlace(FloatMatrix right, bool parallel = false, int? newStride = null)
    {
        if (_cols != right._rows)
        {
            throw new ArgumentException($"Incompatible shapes: this={Rows}x{Cols}, right={right.Rows}x{right.Cols}.");
        }

        int m = _rows;
        int n = right._cols;
        int k = _cols;

        int s = newStride ?? n;
        if (s < n) throw new ArgumentOutOfRangeException(nameof(newStride), "newStride must be >= new Cols.");

        float[] newBuffer = new float[checked(m * s)];

        FloatGemm.Multiply(
            _buffer, _stride,
            right._buffer, right._stride,
            newBuffer, s,
            m, n, k,
            parallel);

        _cols = n;
        _stride = s;
        _buffer = newBuffer;
    }

    public void MultiplyFrom(FloatMatrix left, FloatMatrix right, bool parallel = false)
    {
        ValidateMulShape(left, right);
        if (_rows != left._rows || _cols != right._cols)
        {
            throw new ArgumentException($"this must be {left.Rows}x{right.Cols}.");
        }

        Array.Clear(_buffer, 0, _buffer.Length);

        FloatGemm.Multiply(
            left._buffer, left._stride,
            right._buffer, right._stride,
            _buffer, _stride,
            left._rows, right._cols, left._cols,
            parallel);
    }

    public FloatMatrix Transpose(int? stride = null)
    {
        var destination = new FloatMatrix(_cols, _rows, stride ?? _rows);
        TransposeInto(destination);
        return destination;
    }

    public void TransposeInto(FloatMatrix destination)
    {
        if (destination._rows != _cols || destination._cols != _rows)
        {
            throw new ArgumentException($"destination must be {_cols}x{_rows}.");
        }

        const int block = 32;

        int srcRows = _rows;
        int srcCols = _cols;
        int srcStride = _stride;
        int dstStride = destination._stride;

        float[] src = _buffer;
        float[] dst = destination._buffer;

        for (int i0 = 0; i0 < srcRows; i0 += block)
        {
            int iMax = System.Math.Min(srcRows, i0 + block);
            for (int j0 = 0; j0 < srcCols; j0 += block)
            {
                int jMax = System.Math.Min(srcCols, j0 + block);

                for (int i = i0; i < iMax; i++)
                {
                    int srcBase = i * srcStride + j0;
                    for (int j = j0; j < jMax; j++)
                    {
                        dst[j * dstStride + i] = src[srcBase + (j - j0)];
                    }
                }
            }
        }
    }

    public void TransposeInPlace(int? newStride = null)
    {
        if (_rows == _cols)
        {
            int n = _rows;
            for (int i = 0; i < n; i++)
            {
                int rowBase = i * _stride;
                for (int j = i + 1; j < n; j++)
                {
                    int idx1 = rowBase + j;
                    int idx2 = j * _stride + i;
                    (_buffer[idx1], _buffer[idx2]) = (_buffer[idx2], _buffer[idx1]);
                }
            }
            return;
        }

        int transposedRows = _cols;
        int transposedCols = _rows;

        int s = newStride ?? transposedCols;
        if (s < transposedCols) throw new ArgumentOutOfRangeException(nameof(newStride), "newStride must be >= new Cols.");

        var destination = new FloatMatrix(transposedRows, transposedCols, s);
        TransposeInto(destination);

        _rows = transposedRows;
        _cols = transposedCols;
        _stride = s;
        _buffer = destination._buffer;
    }

    public FloatMatrix SliceColumns(int startCol, int endExclusiveCol)
    {
        if (startCol < 0 || startCol > endExclusiveCol || endExclusiveCol > _cols)
        {
            throw new ArgumentOutOfRangeException(nameof(startCol), "Invalid slice column range.");
        }

        int newCols = endExclusiveCol - startCol;
        var result = new FloatMatrix(_rows, newCols);
        if (_rows == 0 || newCols == 0)
        {
            return result;
        }

        for (int row = 0; row < _rows; row++)
        {
            RowSpan(row).Slice(startCol, newCols).CopyTo(result.RowSpan(row));
        }

        return result;
    }

    public static FloatMatrix ConcatColumns(FloatMatrix left, FloatMatrix right)
    {
        if (left._rows != right._rows)
        {
            throw new ArgumentException($"Row counts must match: left={left._rows}, right={right._rows}.");
        }

        int rows = left._rows;
        int leftCols = left._cols;
        int rightCols = right._cols;
        var result = new FloatMatrix(rows, leftCols + rightCols);
        if (rows == 0)
        {
            return result;
        }

        for (int row = 0; row < rows; row++)
        {
            var dst = result.RowSpan(row);
            left.RowSpan(row).CopyTo(dst[..leftCols]);
            right.RowSpan(row).CopyTo(dst[leftCols..]);
        }

        return result;
    }

    private void CopyFromArray(float[,] source)
    {
        if (_rows == 0 || _cols == 0)
        {
            return;
        }

        if (_stride == _cols)
        {
            MemoryMarshal.CreateReadOnlySpan(ref source[0, 0], _rows * _cols).CopyTo(_buffer.AsSpan(0, _rows * _cols));
        }
        else
        {
            for (int row = 0; row < _rows; row++)
            {
                MemoryMarshal.CreateReadOnlySpan(ref source[row, 0], _cols).CopyTo(RowSpan(row));
            }
        }
    }

    private static void MultiplyIntoCore(FloatMatrix a, FloatMatrix b, FloatMatrix c, bool parallel, int maxDegreeOfParallelism)
    {
        Array.Clear(c._buffer, 0, c._buffer.Length);

        FloatGemm.Multiply(
            a._buffer, a._stride,
            b._buffer, b._stride,
            c._buffer, c._stride,
            a._rows, b._cols, a._cols,
            parallel,
            maxDegreeOfParallelism);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddRow(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
    {
        int simd = Vector<float>.Count;
        int i = 0;
        for (; i <= left.Length - simd; i += simd)
        {
            (new Vector<float>(left.Slice(i, simd)) + new Vector<float>(right.Slice(i, simd))).CopyTo(destination.Slice(i, simd));
        }

        for (; i < left.Length; i++)
        {
            destination[i] = left[i] + right[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SubtractRow(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
    {
        int simd = Vector<float>.Count;
        int i = 0;
        for (; i <= left.Length - simd; i += simd)
        {
            (new Vector<float>(left.Slice(i, simd)) - new Vector<float>(right.Slice(i, simd))).CopyTo(destination.Slice(i, simd));
        }

        for (; i < left.Length; i++)
        {
            destination[i] = left[i] - right[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleRow(ReadOnlySpan<float> source, Span<float> destination, float factor)
    {
        int simd = Vector<float>.Count;
        var scalar = new Vector<float>(factor);
        int i = 0;
        for (; i <= source.Length - simd; i += simd)
        {
            (new Vector<float>(source.Slice(i, simd)) * scalar).CopyTo(destination.Slice(i, simd));
        }

        for (; i < source.Length; i++)
        {
            destination[i] = source[i] * factor;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ShouldParallelize(int rows, int cols) =>
        rows > 1 && (long)rows * cols >= ElementwiseParallelThreshold;

    private void EnsureCompactLayout()
    {
        if (!IsCompact)
        {
            throw new InvalidOperationException("This operation requires compact storage (stride == cols).");
        }
    }

    private static void ValidateShape(int rows, int cols)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
    }

    private static void ValidateSameShape(FloatMatrix left, FloatMatrix right)
    {
        if (left._rows != right._rows || left._cols != right._cols)
        {
            throw new ArgumentException($"Shape mismatch: left={left._rows}x{left._cols}, right={right._rows}x{right._cols}.");
        }
    }

    private static void ValidateMulShape(FloatMatrix a, FloatMatrix b)
    {
        if (a._cols != b._rows)
        {
            throw new ArgumentException($"Incompatible shapes: A={a._rows}x{a._cols}, B={b._rows}x{b._cols}.");
        }
    }

    private static void ValidateMulDestination(FloatMatrix a, FloatMatrix b, FloatMatrix c)
    {
        if (c._rows != a._rows || c._cols != b._cols)
        {
            throw new ArgumentException($"C must be {a._rows}x{b._cols}.");
        }
    }

    private static class FloatGemm
    {
        private const int BlockK = 32;
        private const long ParallelWorkThreshold = 2_000_000;

        public static void Multiply(
            float[] a, int strideA,
            float[] b, int strideB,
            float[] c, int strideC,
            int M, int N, int K,
            bool parallel,
            int maxDegreeOfParallelism = -1)
        {
            if (M <= 0 || N <= 0 || K <= 0) return;

            int vw = Vector<float>.Count;
            int n3 = vw * 3;
            int nEnd3 = (N / n3) * n3;
            int nEnd1 = (N / vw) * vw;

            int chunkRows = 32;
            if (chunkRows > M) chunkRows = 4;
            chunkRows = (chunkRows / 4) * 4;

            int chunkCount = (M + chunkRows - 1) / chunkRows;

            long work = (long)M * N * K;
            bool doParallel = parallel && chunkCount > 1 && work >= ParallelWorkThreshold;

            if (doParallel)
            {
                if (maxDegreeOfParallelism > 0)
                {
                    var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
                    Parallel.For(0, chunkCount, options, chunkIdx =>
                    {
                        int iStart = chunkIdx * chunkRows;
                        int iEnd = System.Math.Min(M, iStart + chunkRows);
                        MultiplyChunk(a, strideA, b, strideB, c, strideC, iStart, iEnd, N, K, nEnd3, nEnd1);
                    });
                }
                else
                {
                    Parallel.For(0, chunkCount, chunkIdx =>
                    {
                        int iStart = chunkIdx * chunkRows;
                        int iEnd = System.Math.Min(M, iStart + chunkRows);
                        MultiplyChunk(a, strideA, b, strideB, c, strideC, iStart, iEnd, N, K, nEnd3, nEnd1);
                    });
                }
            }
            else
            {
                MultiplyChunk(a, strideA, b, strideB, c, strideC, 0, M, N, K, nEnd3, nEnd1);
            }
        }

        private static void MultiplyChunk(
            float[] a, int strideA,
            float[] b, int strideB,
            float[] c, int strideC,
            int iStart, int iEnd,
            int N, int K,
            int nEnd3, int nEnd1)
        {
            ref float a0 = ref a[0];
            ref float b0 = ref b[0];
            ref float c0 = ref c[0];

            int vw = Vector<float>.Count;
            int n3 = vw * 3;

            for (int k0 = 0; k0 < K; k0 += BlockK)
            {
                int kk = K - k0;
                if (kk > BlockK) kk = BlockK;

                ref float bPanel0 = ref Unsafe.Add(ref b0, k0 * strideB);

                int i = iStart;

                for (; i + 3 < iEnd; i += 4)
                {
                    ref float aBlock = ref Unsafe.Add(ref a0, i * strideA + k0);
                    ref float cRow0 = ref Unsafe.Add(ref c0, i * strideC);

                    int j = 0;
                    for (; j < nEnd3; j += n3)
                    {
                        ref float bBlock = ref Unsafe.Add(ref bPanel0, j);
                        ref float cBlock = ref Unsafe.Add(ref cRow0, j);
                        Kernel4x3(kk, in aBlock, strideA, in bBlock, strideB, ref cBlock, strideC);
                    }
                    for (; j < nEnd1; j += vw)
                    {
                        ref float bBlock = ref Unsafe.Add(ref bPanel0, j);
                        ref float cBlock = ref Unsafe.Add(ref cRow0, j);
                        Kernel4x1(kk, in aBlock, strideA, in bBlock, strideB, ref cBlock, strideC);
                    }
                    if (j < N)
                    {
                        ScalarTail4Rows(kk, i, j, N, k0, strideA, strideB, strideC, ref a0, ref b0, ref c0);
                    }
                }

                for (; i < iEnd; i++)
                {
                    ref float aRow = ref Unsafe.Add(ref a0, i * strideA + k0);
                    ref float cRow = ref Unsafe.Add(ref c0, i * strideC);

                    int j = 0;
                    for (; j < nEnd3; j += n3)
                    {
                        ref float bBlock = ref Unsafe.Add(ref bPanel0, j);
                        ref float cBlock = ref Unsafe.Add(ref cRow, j);
                        Kernel1x3(kk, in aRow, in bBlock, strideB, ref cBlock);
                    }
                    for (; j < nEnd1; j += vw)
                    {
                        ref float bBlock = ref Unsafe.Add(ref bPanel0, j);
                        ref float cBlock = ref Unsafe.Add(ref cRow, j);
                        Kernel1x1(kk, in aRow, in bBlock, strideB, ref cBlock);
                    }
                    if (j < N)
                    {
                        ScalarTail1Row(kk, i, j, N, k0, strideA, strideB, strideC, ref a0, ref b0, ref c0);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ScalarTail4Rows(
            int kk,
            int i, int jStart, int N,
            int k0,
            int strideA, int strideB, int strideC,
            ref float a0, ref float b0, ref float c0)
        {
            int aRow0 = (i + 0) * strideA + k0;
            int aRow1 = (i + 1) * strideA + k0;
            int aRow2 = (i + 2) * strideA + k0;
            int aRow3 = (i + 3) * strideA + k0;

            int cRow0 = (i + 0) * strideC;
            int cRow1 = (i + 1) * strideC;
            int cRow2 = (i + 2) * strideC;
            int cRow3 = (i + 3) * strideC;

            for (int j = jStart; j < N; j++)
            {
                float c0v = Unsafe.Add(ref c0, cRow0 + j);
                float c1v = Unsafe.Add(ref c0, cRow1 + j);
                float c2v = Unsafe.Add(ref c0, cRow2 + j);
                float c3v = Unsafe.Add(ref c0, cRow3 + j);

                ref float bRef = ref Unsafe.Add(ref b0, k0 * strideB + j);
                for (int k = 0; k < kk; k++)
                {
                    float bv = bRef;
                    c0v += Unsafe.Add(ref a0, aRow0 + k) * bv;
                    c1v += Unsafe.Add(ref a0, aRow1 + k) * bv;
                    c2v += Unsafe.Add(ref a0, aRow2 + k) * bv;
                    c3v += Unsafe.Add(ref a0, aRow3 + k) * bv;
                    bRef = ref Unsafe.Add(ref bRef, strideB);
                }

                Unsafe.Add(ref c0, cRow0 + j) = c0v;
                Unsafe.Add(ref c0, cRow1 + j) = c1v;
                Unsafe.Add(ref c0, cRow2 + j) = c2v;
                Unsafe.Add(ref c0, cRow3 + j) = c3v;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ScalarTail1Row(
            int kk,
            int i, int jStart, int N,
            int k0,
            int strideA, int strideB, int strideC,
            ref float a0, ref float b0, ref float c0)
        {
            int aRow = i * strideA + k0;
            int cRow = i * strideC;

            for (int j = jStart; j < N; j++)
            {
                float cv = Unsafe.Add(ref c0, cRow + j);

                ref float bRef = ref Unsafe.Add(ref b0, k0 * strideB + j);
                for (int k = 0; k < kk; k++)
                {
                    cv += Unsafe.Add(ref a0, aRow + k) * bRef;
                    bRef = ref Unsafe.Add(ref bRef, strideB);
                }

                Unsafe.Add(ref c0, cRow + j) = cv;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel4x3(
            int K,
            in float A, int strideA,
            in float B, int strideB,
            ref float C, int strideC)
        {
            int vw = Vector<float>.Count;

            ref float cBase = ref C;
            Vector<float> c00 = Load(ref cBase);
            Vector<float> c01 = Load(ref Unsafe.Add(ref cBase, vw));
            Vector<float> c02 = Load(ref Unsafe.Add(ref cBase, 2 * vw));

            ref float c1Base = ref Unsafe.Add(ref cBase, strideC);
            Vector<float> c10 = Load(ref c1Base);
            Vector<float> c11 = Load(ref Unsafe.Add(ref c1Base, vw));
            Vector<float> c12 = Load(ref Unsafe.Add(ref c1Base, 2 * vw));

            ref float c2Base = ref Unsafe.Add(ref c1Base, strideC);
            Vector<float> c20 = Load(ref c2Base);
            Vector<float> c21 = Load(ref Unsafe.Add(ref c2Base, vw));
            Vector<float> c22 = Load(ref Unsafe.Add(ref c2Base, 2 * vw));

            ref float c3Base = ref Unsafe.Add(ref c2Base, strideC);
            Vector<float> c30 = Load(ref c3Base);
            Vector<float> c31 = Load(ref Unsafe.Add(ref c3Base, vw));
            Vector<float> c32 = Load(ref Unsafe.Add(ref c3Base, 2 * vw));

            ref float pALine = ref Unsafe.AsRef(in A);
            ref float pB = ref Unsafe.AsRef(in B);

            for (int k = 0; k < K; k++)
            {
                Vector<float> b0 = Load(ref pB);
                Vector<float> b1 = Load(ref Unsafe.Add(ref pB, vw));
                Vector<float> b2 = Load(ref Unsafe.Add(ref pB, 2 * vw));

                ref float pA = ref pALine;

                Vector<float> va0 = new Vector<float>(pA);
                c00 = Fma(va0, b0, c00);
                c01 = Fma(va0, b1, c01);
                c02 = Fma(va0, b2, c02);
                pA = ref Unsafe.Add(ref pA, strideA);

                Vector<float> va1 = new Vector<float>(pA);
                c10 = Fma(va1, b0, c10);
                c11 = Fma(va1, b1, c11);
                c12 = Fma(va1, b2, c12);
                pA = ref Unsafe.Add(ref pA, strideA);

                Vector<float> va2 = new Vector<float>(pA);
                c20 = Fma(va2, b0, c20);
                c21 = Fma(va2, b1, c21);
                c22 = Fma(va2, b2, c22);
                pA = ref Unsafe.Add(ref pA, strideA);

                Vector<float> va3 = new Vector<float>(pA);
                c30 = Fma(va3, b0, c30);
                c31 = Fma(va3, b1, c31);
                c32 = Fma(va3, b2, c32);

                pALine = ref Unsafe.Add(ref pALine, 1);
                pB = ref Unsafe.Add(ref pB, strideB);
            }

            Store(c00, ref cBase);
            Store(c01, ref Unsafe.Add(ref cBase, vw));
            Store(c02, ref Unsafe.Add(ref cBase, 2 * vw));

            Store(c10, ref c1Base);
            Store(c11, ref Unsafe.Add(ref c1Base, vw));
            Store(c12, ref Unsafe.Add(ref c1Base, 2 * vw));

            Store(c20, ref c2Base);
            Store(c21, ref Unsafe.Add(ref c2Base, vw));
            Store(c22, ref Unsafe.Add(ref c2Base, 2 * vw));

            Store(c30, ref c3Base);
            Store(c31, ref Unsafe.Add(ref c3Base, vw));
            Store(c32, ref Unsafe.Add(ref c3Base, 2 * vw));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel4x1(
            int K,
            in float A, int strideA,
            in float B, int strideB,
            ref float C, int strideC)
        {
            Vector<float> c0 = Load(ref C);
            ref float c1Ref = ref Unsafe.Add(ref C, strideC);
            Vector<float> c1 = Load(ref c1Ref);
            ref float c2Ref = ref Unsafe.Add(ref c1Ref, strideC);
            Vector<float> c2 = Load(ref c2Ref);
            ref float c3Ref = ref Unsafe.Add(ref c2Ref, strideC);
            Vector<float> c3 = Load(ref c3Ref);

            ref float pALine = ref Unsafe.AsRef(in A);
            ref float pB = ref Unsafe.AsRef(in B);

            for (int k = 0; k < K; k++)
            {
                Vector<float> bv = Load(ref pB);

                ref float pA = ref pALine;
                c0 = Fma(new Vector<float>(pA), bv, c0); pA = ref Unsafe.Add(ref pA, strideA);
                c1 = Fma(new Vector<float>(pA), bv, c1); pA = ref Unsafe.Add(ref pA, strideA);
                c2 = Fma(new Vector<float>(pA), bv, c2); pA = ref Unsafe.Add(ref pA, strideA);
                c3 = Fma(new Vector<float>(pA), bv, c3);

                pALine = ref Unsafe.Add(ref pALine, 1);
                pB = ref Unsafe.Add(ref pB, strideB);
            }

            Store(c0, ref C);
            Store(c1, ref c1Ref);
            Store(c2, ref c2Ref);
            Store(c3, ref c3Ref);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel1x3(
            int K,
            in float ARow,
            in float B, int strideB,
            ref float CRow)
        {
            int vw = Vector<float>.Count;

            ref float cBase = ref CRow;
            Vector<float> c0 = Load(ref cBase);
            Vector<float> c1 = Load(ref Unsafe.Add(ref cBase, vw));
            Vector<float> c2 = Load(ref Unsafe.Add(ref cBase, 2 * vw));

            ref float pA = ref Unsafe.AsRef(in ARow);
            ref float pB = ref Unsafe.AsRef(in B);

            for (int k = 0; k < K; k++)
            {
                Vector<float> b0 = Load(ref pB);
                Vector<float> b1 = Load(ref Unsafe.Add(ref pB, vw));
                Vector<float> b2 = Load(ref Unsafe.Add(ref pB, 2 * vw));

                Vector<float> va = new Vector<float>(pA);
                c0 = Fma(va, b0, c0);
                c1 = Fma(va, b1, c1);
                c2 = Fma(va, b2, c2);

                pA = ref Unsafe.Add(ref pA, 1);
                pB = ref Unsafe.Add(ref pB, strideB);
            }

            Store(c0, ref cBase);
            Store(c1, ref Unsafe.Add(ref cBase, vw));
            Store(c2, ref Unsafe.Add(ref cBase, 2 * vw));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Kernel1x1(
            int K,
            in float ARow,
            in float B, int strideB,
            ref float CRow)
        {
            Vector<float> c0 = Load(ref CRow);

            ref float pA = ref Unsafe.AsRef(in ARow);
            ref float pB = ref Unsafe.AsRef(in B);

            for (int k = 0; k < K; k++)
            {
                Vector<float> bv = Load(ref pB);
                c0 = Fma(new Vector<float>(pA), bv, c0);

                pA = ref Unsafe.Add(ref pA, 1);
                pB = ref Unsafe.Add(ref pB, strideB);
            }

            Store(c0, ref CRow);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> Fma(Vector<float> left, Vector<float> right, Vector<float> addend)
            => Vector.FusedMultiplyAdd(left, right, addend);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> Load(ref float source)
            => Vector.LoadUnsafe(in source);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store(Vector<float> source, ref float destination)
            => Vector.StoreUnsafe(source, ref destination);
    }
}
