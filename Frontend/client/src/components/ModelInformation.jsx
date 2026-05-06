import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { RefreshCw, Activity, Database, Settings, AlertCircle, CheckCircle2, XCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { apiClient } from "@/lib/apiClient";

export function ModelInformation() {
  const [modelInfo, setModelInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const { toast } = useToast();

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [modelData, healthData] = await Promise.all([
        apiClient.getModelInfo(),
        apiClient.getHealth()
      ]);
      
      setModelInfo(modelData);
      setHealthStatus(healthData);
      
      toast({
        title: "Data Refreshed",
        description: "Model information and health status updated",
      });
    } catch (err) {
      setError(err.message);
      toast({
        title: "Refresh Failed",
        description: err.message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case "healthy":
        return <CheckCircle2 className="h-4 w-4 text-green-600" />;
      case "degraded":
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
      case "unhealthy":
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusBadgeVariant = (status) => {
    switch (status) {
      case "healthy":
        return "default";
      case "degraded":
        return "secondary";
      case "unhealthy":
        return "destructive";
      default:
        return "outline";
    }
  };

  const formatUptime = (uptime) => {
    return uptime || "N/A";
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header with Refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Model Information</h2>
          <p className="text-muted-foreground">Real-time API status and model details</p>
        </div>
        <Button 
          onClick={fetchData}
          disabled={isLoading}
          variant="outline"
          data-testid="button-refresh"
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* API Health Status */}
      {healthStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              API Health Status
            </CardTitle>
            <CardDescription>
              Real-time monitoring of API endpoints and system performance
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(healthStatus?.status || 'unknown')}
                <span className="font-medium">Overall Status</span>
              </div>
              <Badge 
                variant={getStatusBadgeVariant(healthStatus?.status || 'unknown')}
                data-testid="badge-health-status"
              >
                {(healthStatus?.status || 'unknown').toUpperCase()}
              </Badge>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center space-y-1">
                <div className="text-lg font-bold" data-testid="text-uptime">{healthStatus?.uptime ? formatUptime(healthStatus.uptime) : 'N/A'}</div>
                <div className="text-sm text-muted-foreground">Uptime</div>
              </div>
              <div className="text-center space-y-1">
                <div className="text-lg font-bold" data-testid="text-response-time">{healthStatus?.response_time_ms ? `${healthStatus.response_time_ms}ms` : 'N/A'}</div>
                <div className="text-sm text-muted-foreground">Response Time</div>
              </div>
              <div className="text-center space-y-1">
                <div className="text-lg font-bold" data-testid="text-requests-today">{healthStatus?.requests_today?.toLocaleString() || '0'}</div>
                <div className="text-sm text-muted-foreground">Requests Today</div>
              </div>
              <div className="text-center space-y-1">
                <div className="text-lg font-bold">{healthStatus?.last_check ? formatDate(healthStatus.last_check) : 'N/A'}</div>
                <div className="text-sm text-muted-foreground">Last Check</div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Memory Usage</span>
                  <span className="text-sm text-muted-foreground">{healthStatus?.memory_usage ? (healthStatus.memory_usage * 100).toFixed(1) : '0'}%</span>
                </div>
                <Progress value={healthStatus?.memory_usage ? healthStatus.memory_usage * 100 : 0} data-testid="progress-memory" />
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">CPU Usage</span>
                  <span className="text-sm text-muted-foreground">{healthStatus?.cpu_usage ? (healthStatus.cpu_usage * 100).toFixed(1) : '0'}%</span>
                </div>
                <Progress value={healthStatus?.cpu_usage ? healthStatus.cpu_usage * 100 : 0} data-testid="progress-cpu" />
              </div>
            </div>

            {healthStatus?.endpoints && (
              <div className="space-y-2">
                <h4 className="font-medium">Endpoint Status</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {Object.entries(healthStatus.endpoints).map(([endpoint, status]) => (
                    <div key={endpoint} className="flex items-center justify-between p-2 bg-muted rounded">
                      <code className="text-sm">{endpoint}</code>
                      <Badge variant={getStatusBadgeVariant(status)} className="text-xs">
                        {status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Model Details */}
      {modelInfo && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Model Details
              </CardTitle>
              <CardDescription>
                Information about the BBB permeability prediction model
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Model Type</Label>
                    <div className="text-lg font-mono" data-testid="text-model-type">{modelInfo?.model_type || 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Version</Label>
                    <div className="text-lg font-mono" data-testid="text-model-version">{modelInfo?.model_version || 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Features Count</Label>
                    <div className="text-lg font-mono" data-testid="text-features-count">{modelInfo?.feature_count?.toLocaleString() || 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Descriptor Library</Label>
                    <div className="text-lg font-mono" data-testid="text-descriptor-library">{modelInfo?.descriptor_library || 'N/A'}</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Training Dataset Size</Label>
                    <div className="text-lg font-mono" data-testid="text-dataset-size">{modelInfo?.training_dataset_size?.toLocaleString() || 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Validation Accuracy</Label>
                    <div className="text-lg font-mono" data-testid="text-validation-accuracy">{modelInfo?.validation_accuracy ? (modelInfo.validation_accuracy * 100).toFixed(2) + '%' : 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Cross-Validation Score</Label>
                    <div className="text-lg font-mono" data-testid="text-cv-score">{modelInfo?.cross_validation_score ? (modelInfo.cross_validation_score * 100).toFixed(2) + '%' : 'N/A'}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Default Threshold</Label>
                    <div className="text-lg font-mono" data-testid="text-default-threshold">{modelInfo?.default_threshold?.toFixed(4) || 'N/A'}</div>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium">Supported Formats</Label>
                <div className="flex flex-wrap gap-2">
                  {modelInfo?.supported_file_types?.map((format) => (
                    <Badge key={format} variant="outline" data-testid={`badge-format-${format.toLowerCase()}`}>
                      {format}
                    </Badge>
                  )) || <span className="text-sm text-muted-foreground">N/A</span>}
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium">Last Updated</Label>
                <div className="text-sm text-muted-foreground" data-testid="text-last-updated">
                  {modelInfo?.last_updated ? formatDate(modelInfo.last_updated) : 'N/A'}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
              <CardDescription>
                Top 5 most important molecular descriptors for BBB prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(modelInfo?.feature_importance_top && Array.isArray(modelInfo.feature_importance_top) && modelInfo.feature_importance_top.length > 0) ? (
                  modelInfo.feature_importance_top.map((feature, index) => (
                    <div key={feature?.name || `feature-${index}`} className="space-y-2" data-testid={`feature-${index}`}>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{feature?.name || `Feature ${index + 1}`}</span>
                        <span className="text-sm text-muted-foreground">
                          {((feature?.importance || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={(feature?.importance || 0) * 100} />
                    </div>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">Feature importance data not available</span>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Environment Information
              </CardTitle>
              <CardDescription>
                Server environment and dependency versions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {modelInfo?.environment ? Object.entries(modelInfo.environment).map(([key, value]) => (
                  <div key={key} className="flex justify-between p-3 bg-muted rounded">
                    <span className="font-medium capitalize">{key.replace('_', ' ')}</span>
                    <code className="text-sm" data-testid={`env-${key}`}>{value}</code>
                  </div>
                )) : <span className="text-sm text-muted-foreground">Environment information not available</span>}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function Label({ children, className = "" }) {
  return <div className={`text-sm font-medium ${className}`}>{children}</div>;
}