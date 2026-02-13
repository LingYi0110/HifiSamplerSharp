using System.Text;
using System.Text.Json;
using HifiSampler.Server.Contracts;

namespace HifiSampler.Server;

public static class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        var samplerPort = int.TryParse(builder.Configuration["Sampler:Port"], out var parsedPort)
            ? parsedPort
            : 8572;
        builder.WebHost.UseUrls($"http://127.0.0.1:{samplerPort}");
        builder.Services.AddOpenApi();
        builder.Services.AddSingleton<ResamplerService>();

        var app = builder.Build();
        _ = app.Services.GetRequiredService<ResamplerService>();

        if (app.Environment.IsDevelopment())
        {
            app.MapOpenApi();
        }

        app.Run(async context =>
        {
            if (context.Request.Path != "/")
            {
                context.Response.StatusCode = StatusCodes.Status404NotFound;
                return;
            }

            var service = context.RequestServices.GetRequiredService<ResamplerService>();
            var logger = context.RequestServices.GetRequiredService<ILogger<ResamplerService>>();

            if (HttpMethods.IsGet(context.Request.Method))
            {
                if (service.IsReady)
                {
                    await WritePlainTextAsync(
                        context,
                        "Server Ready",
                        StatusCodes.Status200OK);
                    return;
                }

                await WritePlainTextAsync(
                    context,
                    "Server Initializing",
                    StatusCodes.Status503ServiceUnavailable);
                return;
            }

            if (!HttpMethods.IsPost(context.Request.Method))
            {
                await WritePlainTextAsync(
                    context,
                    "Method Not Allowed",
                    StatusCodes.Status405MethodNotAllowed);
                return;
            }

            if (!service.IsReady)
            {
                await WritePlainTextAsync(
                    context,
                    "Server initializing, please retry.",
                    StatusCodes.Status503ServiceUnavailable);
                return;
            }

            ResamplerRequestDto? request;
            try
            {
                request = await JsonSerializer.DeserializeAsync(
                    context.Request.Body,
                    ServerJsonContext.Default.ResamplerRequestDto,
                    context.RequestAborted);
            }
            catch (Exception ex)
            {
                await WritePlainTextAsync(
                    context,
                    $"Error processing: Invalid JSON.\n{ex}",
                    StatusCodes.Status500InternalServerError);
                return;
            }

            if (request is null)
            {
                await WritePlainTextAsync(
                    context,
                    "Error processing: Empty request body.",
                    StatusCodes.Status500InternalServerError);
                return;
            }

            logger.LogInformation(
                "Incoming HTTP resample request from {RemoteIp}",
                context.Connection.RemoteIpAddress?.ToString() ?? "unknown");

            var result = await service.RenderAsync(request, context.RequestAborted);
            var payload = result.Traceback is null ? result.Message : $"{result.Message}\n{result.Traceback}";
            await WritePlainTextAsync(context, payload, result.StatusCode);
        });

        app.Run();
    }

    private static async Task WritePlainTextAsync(HttpContext context, string content, int statusCode)
    {
        context.Response.StatusCode = statusCode;
        context.Response.ContentType = "text/plain; charset=utf-8";
        await context.Response.WriteAsync(content, Encoding.UTF8);
    }
}
