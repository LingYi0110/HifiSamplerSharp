using System.Text;
using System.Text.Json;
using HifiSampler.Server.Contracts;

namespace HifiSampler.Server;

public static class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        var samplerPort = builder.Configuration.GetValue<int?>("Sampler:Port") ?? 8572;
        builder.WebHost.UseUrls($"http://127.0.0.1:{samplerPort}");
        builder.Services.AddOpenApi();
        builder.Services.AddSingleton<ResamplerService>();

        var app = builder.Build();
        _ = app.Services.GetRequiredService<ResamplerService>();

        if (app.Environment.IsDevelopment())
        {
            app.MapOpenApi();
        }

        app.MapGet("/", (ResamplerService service) =>
            service.IsReady
                ? Results.Text("Server Ready", "text/plain", Encoding.UTF8, StatusCodes.Status200OK)
                : Results.Text("Server Initializing", "text/plain", Encoding.UTF8, StatusCodes.Status503ServiceUnavailable));

        app.MapPost("/", async (HttpContext context, ResamplerService service, CancellationToken cancellationToken) =>
        {
            if (!service.IsReady)
            {
                return Results.Text("Server initializing, please retry.", "text/plain", Encoding.UTF8,
                    StatusCodes.Status503ServiceUnavailable);
            }

            ResamplerRequestDto? request;
            try
            {
                request = await JsonSerializer.DeserializeAsync<ResamplerRequestDto>(
                    context.Request.Body,
                    new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    },
                    cancellationToken);
            }
            catch (Exception ex)
            {
                return Results.Text($"Error processing: Invalid JSON.\n{ex}", "text/plain", Encoding.UTF8,
                    StatusCodes.Status500InternalServerError);
            }

            if (request is null)
            {
                return Results.Text("Error processing: Empty request body.", "text/plain", Encoding.UTF8,
                    StatusCodes.Status500InternalServerError);
            }

            var result = await service.RenderAsync(request, cancellationToken);
            var payload = result.Traceback is null ? result.Message : $"{result.Message}\n{result.Traceback}";
            return Results.Text(payload, "text/plain", Encoding.UTF8, result.StatusCode);
        });

        app.Run();
    }
}
