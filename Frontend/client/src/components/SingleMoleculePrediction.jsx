import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { AlertCircle, CheckCircle2, XCircle, Loader2, Beaker, RotateCcw, FlaskConical } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { apiClient } from "@/lib/apiClient";

// JSME Configuration
const JSME_CONFIG = {
  menu: true,
  depict: false,
  highlight: true,
  query: false,
  nostatusbar: true
};

const CAFFEINE_SMILES = 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C';

export function SingleMoleculePrediction() {
  const [smiles, setSmiles] = useState("");
  const [moleculeName, setMoleculeName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const { toast } = useToast();

  // JSME Editor state
  const [useEditor, setUseEditor] = useState(false);
  const [jsmeInstance, setJsmeInstance] = useState(null);
  const [editorError, setEditorError] = useState(null);
  const [jsmeLoaded, setJsmeLoaded] = useState(false);
  const jsmeRef = useRef(null);
  const initRef = useRef(false);

  // Load JSME script on component mount
  useEffect(() => {
    const loadJSME = () => {
      return new Promise((resolve, reject) => {
        if (window.JSApplet) {
          setJsmeLoaded(true);
          resolve();
          return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js';
        script.onload = () => {
          setTimeout(() => {
            setJsmeLoaded(true);
            resolve();
          }, 100);
        };
        script.onerror = () => reject(new Error('JSME script failed to load. Please check your network connection.'));
        document.head.appendChild(script);
      });
    };

    loadJSME().catch(error => {
      console.error('Failed to load JSME:', error);
      setEditorError('Failed to load chemical editor library.');
    });
  }, []);

  // Initialize JSME editor when editor is enabled
  useEffect(() => {
    const initializeJSME = async () => {
      if (initRef.current || !useEditor || !jsmeRef.current || !jsmeLoaded) return;
      
      try {
        if (!window.JSApplet) {
          setEditorError('JSME library not loaded. Please refresh the page.');
          return;
        }

        initRef.current = true;
        jsmeRef.current.innerHTML = '';
        
        const jsme = new window.JSApplet.JSME("jsme-container", "100%", "300px", JSME_CONFIG);
        setJsmeInstance(jsme);
        setEditorError(null);

        // Set callback for structure changes
        jsme.setCallBack("AfterStructureModified", function (jsmeEvent) {
          const smilesString = jsme.smiles();
          setSmiles(smilesString);
        });

        // Load existing SMILES if any
        if (smiles) {
          jsme.readGenericMolecularInput(smiles);
        }

      } catch (error) {
        console.error('Failed to initialize JSME:', error);
        setEditorError('Failed to initialize chemical editor: ' + error.message);
        initRef.current = false;
      }
    };

    if (useEditor && jsmeLoaded) {
      initializeJSME();
    }

    return () => {
      if (jsmeRef.current && !useEditor) {
        jsmeRef.current.innerHTML = '';
        initRef.current = false;
        setJsmeInstance(null);
      }
    };
  }, [useEditor, jsmeLoaded]);

  const clearStructure = () => {
    if (jsmeInstance) {
      jsmeInstance.clear();
      setSmiles('');
    }
  };

  const loadSampleMolecule = () => {
    if (jsmeInstance) {
      jsmeInstance.readGenericMolecularInput(CAFFEINE_SMILES);
      setSmiles(CAFFEINE_SMILES);
      setMoleculeName('Caffeine');
    }
  };

  const handlePredict = async () => {
    if (!smiles.trim()) {
      toast({
        title: "Error",
        description: "Please enter a SMILES string",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiClient.predictSingle(smiles, moleculeName, 0.5228);
      setPrediction(result);
      toast({
        title: "Prediction Complete",
        description: `Result: ${result?.prediction || 'Unknown'}`,
      });
    } catch (err) {
      setError(err.message);
      toast({
        title: "Prediction Failed",
        description: err.message,
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return { level: "High", color: "bg-green-500" };
    if (confidence >= 0.6) return { level: "Medium", color: "bg-yellow-500" };
    return { level: "Low", color: "bg-red-500" };
  };

  const getPredictionIcon = (pred) => {
    return pred === "BBB+" ? (
      <CheckCircle2 className="h-5 w-5 text-green-600" />
    ) : (
      <XCircle className="h-5 w-5 text-red-600" />
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Beaker className="h-5 w-5" />
            Single Molecule Prediction
          </CardTitle>
          <CardDescription>
            Draw chemical structures or enter SMILES strings to predict Blood-Brain Barrier permeability
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          
          {/* Editor Toggle */}
          <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
            <div className="space-y-1">
              <Label className="text-sm font-medium">Chemical Structure Editor</Label>
              <p className="text-xs text-muted-foreground">
                Use the interactive editor to draw molecular structures
              </p>
            </div>
            <Switch
              checked={useEditor}
              onCheckedChange={setUseEditor}
              disabled={!jsmeLoaded}
            />
          </div>

          {/* JSME Editor */}
          {useEditor && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <FlaskConical className="h-4 w-4" />
                  Structure Editor
                </CardTitle>
                {!jsmeLoaded && (
                  <CardDescription className="flex items-center gap-2 text-yellow-600">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Loading chemical editor...
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-4">
                {editorError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{editorError}</AlertDescription>
                  </Alert>
                )}
                
                <div 
                  id="jsme-container" 
                  ref={jsmeRef} 
                  className="w-full h-[300px] border rounded-lg overflow-hidden bg-white"
                  style={{ minHeight: '300px' }}
                />
                
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={clearStructure}
                    disabled={!jsmeInstance}
                  >
                    <RotateCcw className="h-3 w-3 mr-1" />
                    Clear
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={loadSampleMolecule}
                    disabled={!jsmeInstance}
                  >
                    Load Sample (Caffeine)
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* SMILES Input */}
          <div className="space-y-2">
            <Label htmlFor="smiles-input">
              SMILES String *
              {useEditor && (
                <span className="text-xs text-muted-foreground ml-2">
                  (Auto-generated from editor)
                </span>
              )}
            </Label>
            <Textarea
              id="smiles-input"
              placeholder="Enter SMILES string (e.g., CCO for ethanol)"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              className="font-mono text-sm"
              rows={3}
              data-testid="input-smiles"
              readOnly={useEditor && jsmeInstance}
            />
          </div>

          {/* Molecule Name */}
          <div className="space-y-2">
            <Label htmlFor="molecule-name">Molecule Name (Optional)</Label>
            <Input
              id="molecule-name"
              placeholder="Enter molecule name"
              value={moleculeName}
              onChange={(e) => setMoleculeName(e.target.value)}
              data-testid="input-molecule-name"
            />
          </div>

          {/* Predict Button */}
          <Button 
            onClick={handlePredict} 
            disabled={isLoading || !smiles.trim()}
            className="w-full"
            data-testid="button-predict"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Predicting...
              </>
            ) : (
              "Predict BBB Permeability"
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results */}
      {prediction && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getPredictionIcon(prediction?.prediction)}
              Prediction Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Prediction</Label>
                <Badge 
                  variant={prediction?.prediction === "BBB+" ? "default" : "destructive"}
                  className="text-sm px-3 py-1"
                  data-testid="badge-prediction"
                >
                  {prediction?.prediction || 'Unknown'}
                </Badge>
              </div>

              <div className="space-y-2">
                <Label>Confidence Level</Label>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" data-testid="badge-confidence">
                    {getConfidenceLevel(prediction?.confidence || 0).level}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {((prediction?.confidence || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Confidence Score</Label>
              <Progress 
                value={(prediction?.confidence || 0) * 100} 
                className="w-full"
                data-testid="progress-confidence"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>BBB+ Probability</Label>
                <div className="text-2xl font-mono" data-testid="text-prob-positive">
                  {((prediction?.probability_bbb_positive || 0) * 100).toFixed(2)}%
                </div>
              </div>

              <div className="space-y-2">
                <Label>BBB- Probability</Label>
                <div className="text-2xl font-mono" data-testid="text-prob-negative">
                  {((prediction?.probability_bbb_negative || 0) * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Interpretation</Label>
              <p className="text-sm text-muted-foreground leading-relaxed" data-testid="text-interpretation">
                {prediction?.interpretation || 'No interpretation available'}
              </p>
            </div>

            <div className="bg-muted p-4 rounded-lg space-y-2">
              <h4 className="font-medium">Molecular Information</h4>
              <div className="space-y-1 text-sm">
                <div><strong>Name:</strong> {prediction?.name || 'N/A'}</div>
                <div><strong>SMILES:</strong> <code className="font-mono text-xs break-all">{prediction?.smiles || 'N/A'}</code></div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}