import React from "react";
import { Brain, Zap, Activity, Github, BookOpen, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="py-16 text-center">
        {/* <div className="mx-auto w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mb-6">
          <Brain className="h-10 w-10 text-primary" />
        </div> */}
        {/* <h1 className="text-4xl font-bold mb-4">BBB Predictor</h1>
        <p className="text-lg text-muted-foreground mb-6">
          Machine learning powered prediction of blood-brain barrier permeability.
        </p>
        <div className="flex flex-wrap justify-center gap-3 mb-6">
          <Badge variant="secondary"><Zap className="w-4 h-4 mr-1" /> 95%+ Accuracy</Badge>
          <Badge variant="secondary"><Activity className="w-4 h-4 mr-1" /> Real-time</Badge>
          <Badge variant="secondary"><BookOpen className="w-4 h-4 mr-1" /> Open Source</Badge>
        </div> */}
        {/* <div className="flex gap-4 justify-center">
          <Button asChild><a href="#get-started">Get Started</a></Button>
          <Button variant="outline" asChild>
            <a
              href="https://github.com/Rajnishphe/LGBM-BBB"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2"
            >
              <Github className="w-4 h-4" /> GitHub
            </a>
          </Button>
        </div> */}
      </section>

      {/* What is BBB Predictor */}
      <section className="py-12 text-center max-w-2xl mx-auto">
        <h2 className="text-2xl font-semibold mb-4">What is BBB Predictor?</h2>
        <p className="text-muted-foreground">
          A simple yet powerful tool that predicts if chemical compounds can cross
          the blood-brain barrier — crucial for drug discovery in neurological diseases.
        </p>
      </section>

      {/* Key Features */}
      {/* <section className="py-12 bg-muted/30">
        <div className="container grid gap-6 md:grid-cols-3">
          <Card>
            <CardHeader><CardTitle>Interactive Editor</CardTitle></CardHeader>
            <CardContent>
              Draw or paste molecules (SMILES) directly in your browser.
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Batch Processing</CardTitle></CardHeader>
            <CardContent>
              Upload multiple compounds (CSV, SDF, MOL) for quick predictions.
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Confidence Scoring</CardTitle></CardHeader>
            <CardContent>
              Get probability scores and confidence levels for each prediction.
            </CardContent>
          </Card>
        </div>
      </section> */}

      {/* Disclaimer */}
      {/* <section className="py-12 max-w-2xl mx-auto text-center">
        <Shield className="h-6 w-6 text-amber-600 mx-auto mb-3" />
        <p className="text-sm text-amber-700">
          BBB Predictor is for research and educational purposes only.  
          Not intended for clinical or medical decision-making.
        </p>
      </section> */}
    </div>
  );
}
